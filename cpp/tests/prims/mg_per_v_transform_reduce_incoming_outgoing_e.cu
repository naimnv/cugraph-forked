/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "property_generator.cuh"

#include <utilities/base_fixture.hpp>
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/high_res_clock.h>
#include <utilities/mg_utilities.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <prims/update_edge_src_dst_property.cuh>

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>

#include <cuco/detail/hash_functions.cuh>
#include <cugraph/edge_partition_view.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <sstream>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/equal.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <gtest/gtest.h>

#include <random>

template <typename T>
struct comparator {
  static constexpr double threshold_ratio{1e-2};
  __host__ __device__ bool operator()(T t1, T t2) const
  {
    if constexpr (std::is_floating_point_v<T>) {
      bool passed = (t1 == t2)  // when t1 == t2 == 0
                    ||
                    (std::abs(t1 - t2) < (std::max(std::abs(t1), std::abs(t2)) * threshold_ratio));
      return passed;
    }
    return t1 == t2;
  }
};

struct result_compare {
  const raft::handle_t& handle_;
  result_compare(raft::handle_t const& handle) : handle_(handle) {}

  template <typename... Args>
  auto operator()(const std::tuple<rmm::device_uvector<Args>...>& t1,
                  const std::tuple<rmm::device_uvector<Args>...>& t2)
  {
    using type = thrust::tuple<Args...>;
    return equality_impl(t1, t2, std::make_index_sequence<thrust::tuple_size<type>::value>());
  }

  template <typename T>
  auto operator()(const rmm::device_uvector<T>& t1, const rmm::device_uvector<T>& t2)
  {
    return thrust::equal(
      handle_.get_thrust_policy(), t1.begin(), t1.end(), t2.begin(), comparator<T>());
  }

 private:
  template <typename T, std::size_t... I>
  auto equality_impl(T& t1, T& t2, std::index_sequence<I...>)
  {
    return (... && (result_compare::operator()(std::get<I>(t1), std::get<I>(t2))));
  }
};

template <typename buffer_type>
buffer_type aggregate(const raft::handle_t& handle, const buffer_type& result)
{
  auto aggregated_result =
    cugraph::allocate_dataframe_buffer<cugraph::dataframe_element_t<buffer_type>>(
      0, handle.get_stream());
  cugraph::transform(result, aggregated_result, [&handle](auto& input, auto& output) {
    output = cugraph::test::device_gatherv(handle, input.data(), input.size());
  });
  return aggregated_result;
}

struct Prims_Usecase {
  bool check_correctness{true};
  bool test_weighted{false};
};

template <typename input_usecase_t>
class Tests_MGPerVTransformReduceIncomingOutgoingE
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MGPerVTransformReduceIncomingOutgoingE() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of per_v_transform_reduce_incoming|outgoing_e primitive
  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename result_t,
            bool store_transposed>
  void run_current_test(Prims_Usecase const& prims_usecase, input_usecase_t const& input_usecase)
  {
    HighResClock hr_clock{};

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_clock.start();
    }
    auto [mg_graph, d_mg_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, true>(
        *handle_, input_usecase, prims_usecase.test_weighted, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto mg_graph_view = mg_graph.view();

    // 2. run MG transform reduce

    const int hash_bin_count = 5;
    const int initial_value  = 4;

    auto property_initial_value =
      cugraph::test::generate<vertex_t, result_t>::initial_value(initial_value);

    auto mg_vertex_prop = cugraph::test::generate<vertex_t, result_t>::vertex_property(
      *handle_, *d_mg_renumber_map_labels, hash_bin_count);
    auto mg_src_prop = cugraph::test::generate<vertex_t, result_t>::src_property(
      *handle_, mg_graph_view, mg_vertex_prop);
    auto mg_dst_prop = cugraph::test::generate<vertex_t, result_t>::dst_property(
      *handle_, mg_graph_view, mg_vertex_prop);

    auto out_result = cugraph::allocate_dataframe_buffer<result_t>(
      mg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());
    auto in_result = cugraph::allocate_dataframe_buffer<result_t>(
      mg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_clock.start();
    }

    per_v_transform_reduce_incoming_e(
      *handle_,
      mg_graph_view,
      mg_src_prop.view(),
      mg_dst_prop.view(),
      [] __device__(auto src, auto dst, weight_t wt, auto src_property, auto dst_property) {
        if (src_property < dst_property) {
          return src_property;
        } else {
          return dst_property;
        }
      },
      property_initial_value,
      cugraph::get_dataframe_buffer_begin(in_result));

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG per_v_transform_reduce_incoming_e took " << elapsed_time * 1e-6 << " s.\n";
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_clock.start();
    }

    per_v_transform_reduce_outgoing_e(
      *handle_,
      mg_graph_view,
      mg_src_prop.view(),
      mg_dst_prop.view(),
      [] __device__(auto src, auto dst, weight_t wt, auto src_property, auto dst_property) {
        if (src_property < dst_property) {
          return src_property;
        } else {
          return dst_property;
        }
      },
      property_initial_value,
      cugraph::get_dataframe_buffer_begin(out_result));

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG per_v_transform_reduce_outgoing_e took " << elapsed_time * 1e-6 << " s.\n";
    }

    // 3. compare SG & MG results

    if (prims_usecase.check_correctness) {
      cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, false> sg_graph(*handle_);
      std::tie(sg_graph, std::ignore) =
        cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, false>(
          *handle_, input_usecase, true, false);
      auto sg_graph_view = sg_graph.view();

      auto sg_vertex_prop = cugraph::test::generate<vertex_t, result_t>::vertex_property(
        *handle_,
        thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_first()),
        thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_last()),
        hash_bin_count);
      auto sg_dst_prop = cugraph::test::generate<vertex_t, result_t>::dst_property(
        *handle_, sg_graph_view, sg_vertex_prop);
      auto sg_src_prop = cugraph::test::generate<vertex_t, result_t>::src_property(
        *handle_, sg_graph_view, sg_vertex_prop);
      result_compare comp{*handle_};

      auto global_out_result = cugraph::allocate_dataframe_buffer<result_t>(
        sg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());
      per_v_transform_reduce_outgoing_e(
        *handle_,
        sg_graph_view,
        sg_src_prop.view(),
        sg_dst_prop.view(),
        [] __device__(auto src, auto dst, weight_t wt, auto src_property, auto dst_property) {
          if (src_property < dst_property) {
            return src_property;
          } else {
            return dst_property;
          }
        },
        property_initial_value,
        cugraph::get_dataframe_buffer_begin(global_out_result));

      auto global_in_result = cugraph::allocate_dataframe_buffer<result_t>(
        sg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());
      per_v_transform_reduce_incoming_e(
        *handle_,
        sg_graph_view,
        sg_src_prop.view(),
        sg_dst_prop.view(),
        [] __device__(auto src, auto dst, weight_t wt, auto src_property, auto dst_property) {
          if (src_property < dst_property) {
            return src_property;
          } else {
            return dst_property;
          }
        },
        property_initial_value,
        cugraph::get_dataframe_buffer_begin(global_in_result));
      auto aggregate_labels      = aggregate(*handle_, *d_mg_renumber_map_labels);
      auto aggregate_out_results = aggregate(*handle_, out_result);
      auto aggregate_in_results  = aggregate(*handle_, in_result);
      if (handle_->get_comms().get_rank() == int{0}) {
        std::tie(std::ignore, aggregate_out_results) =
          cugraph::test::sort_by_key(*handle_, aggregate_labels, aggregate_out_results);
        std::tie(std::ignore, aggregate_in_results) =
          cugraph::test::sort_by_key(*handle_, aggregate_labels, aggregate_in_results);
        ASSERT_TRUE(comp(aggregate_out_results, global_out_result));
        ASSERT_TRUE(comp(aggregate_in_results, global_in_result));
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t>
  Tests_MGPerVTransformReduceIncomingOutgoingE<input_usecase_t>::handle_ = nullptr;

using Tests_MGPerVTransformReduceIncomingOutgoingE_File =
  Tests_MGPerVTransformReduceIncomingOutgoingE<cugraph::test::File_Usecase>;
using Tests_MGPerVTransformReduceIncomingOutgoingE_Rmat =
  Tests_MGPerVTransformReduceIncomingOutgoingE<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGPerVTransformReduceIncomingOutgoingE_File,
       CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>, false>(std::get<0>(param),
                                                                              std::get<1>(param));
}

TEST_P(Tests_MGPerVTransformReduceIncomingOutgoingE_Rmat,
       CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVTransformReduceIncomingOutgoingE_File,
       CheckInt32Int32FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>, true>(std::get<0>(param),
                                                                             std::get<1>(param));
}

TEST_P(Tests_MGPerVTransformReduceIncomingOutgoingE_Rmat,
       CheckInt32Int32FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVTransformReduceIncomingOutgoingE_File, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGPerVTransformReduceIncomingOutgoingE_Rmat, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVTransformReduceIncomingOutgoingE_File, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGPerVTransformReduceIncomingOutgoingE_Rmat, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGPerVTransformReduceIncomingOutgoingE_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGPerVTransformReduceIncomingOutgoingE_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{true}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_large_test,
  Tests_MGPerVTransformReduceIncomingOutgoingE_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{false}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
