/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#pragma once

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <detail/graph_partition_utils.cuh>

#include <raft/core/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#ifndef NO_CUGRAPH_OPS
#include <cugraph-ops/graph/sampling.hpp>
#endif

#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <chrono>
#include <cstdlib>
#include <iostream>

namespace cugraph {
template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
rmm::device_uvector<vertex_t> select_random_vertices(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<raft::device_span<vertex_t const>> given_set,
  raft::random::RngState& rng_state,
  size_t select_count,
  bool with_replacement,
  bool sort_vertices,
  bool do_expensive_check)
{
  size_t num_of_elements_in_given_set{0};
  if (given_set) {
    if (do_expensive_check) {
      CUGRAPH_EXPECTS(static_cast<size_t>(thrust::count_if(
                        handle.get_thrust_policy(),
                        (*given_set).begin(),
                        (*given_set).begin() + (*given_set).size(),
                        detail::check_out_of_range_t<vertex_t>{
                          graph_view.local_vertex_partition_range_first(),
                          graph_view.local_vertex_partition_range_last()})) == size_t{0},
                      "Invalid input argument: vertex IDs in the given set must be within vertex "
                      "partition assigned to this GPU");
    }
    num_of_elements_in_given_set = static_cast<size_t>((*given_set).size());
    if (multi_gpu) {
      num_of_elements_in_given_set = host_scalar_allreduce(handle.get_comms(),
                                                           num_of_elements_in_given_set,
                                                           raft::comms::op_t::SUM,
                                                           handle.get_stream());
    }
    CUGRAPH_EXPECTS(
      with_replacement || select_count <= num_of_elements_in_given_set,
      "Invalid input arguments: select_count should not exceed the number of given vertices if "
      "with_replacement == false.");
  } else {
    CUGRAPH_EXPECTS(
      with_replacement || select_count <= static_cast<size_t>(graph_view.number_of_vertices()),
      "Invalid input arguments: select_count should not exceed the number of vertices if "
      "with_replacement == false.");
  }

  rmm::device_uvector<vertex_t> mg_sample_buffer(0, handle.get_stream());

  size_t this_gpu_select_count{0};
  if constexpr (multi_gpu) {
    auto const comm_rank = handle.get_comms().get_rank();
    auto const comm_size = handle.get_comms().get_size();

    this_gpu_select_count =
      select_count / static_cast<size_t>(comm_size) +
      (static_cast<size_t>(comm_rank) < static_cast<size_t>(select_count % comm_size) ? size_t{1}
                                                                                      : size_t{0});
  } else {
    this_gpu_select_count = select_count;
  }

  std::vector<vertex_t> partition_range_lasts;
  if constexpr (multi_gpu) {
    partition_range_lasts = given_set ? cugraph::partition_manager::compute_partition_range_lasts(
                                          handle, static_cast<vertex_t>((*given_set).size()))
                                      : graph_view.vertex_partition_range_lasts();
  }

  if (with_replacement) {
    mg_sample_buffer.resize(this_gpu_select_count, handle.get_stream());
    cugraph::detail::uniform_random_fill(handle.get_stream(),
                                         mg_sample_buffer.data(),
                                         mg_sample_buffer.size(),
                                         vertex_t{0},
                                         given_set
                                           ? static_cast<vertex_t>(num_of_elements_in_given_set)
                                           : graph_view.number_of_vertices(),
                                         rng_state);
  } else {
    vertex_t start_idx;
    size_t mg_buffer_size;
    if (given_set) {
      mg_buffer_size = (*given_set).size();
      start_idx      = vertex_t{0};
      if constexpr (multi_gpu) {
        auto const comm_rank = handle.get_comms().get_rank();
        start_idx            = comm_rank ? partition_range_lasts[comm_rank - 1] : 0;
      }
    } else {
      mg_buffer_size = graph_view.local_vertex_partition_range_size();
      start_idx      = graph_view.local_vertex_partition_range_first();
    }

    mg_sample_buffer = rmm::device_uvector<vertex_t>(mg_buffer_size, handle.get_stream());
    thrust::sequence(
      handle.get_thrust_policy(), mg_sample_buffer.begin(), mg_sample_buffer.end(), start_idx);

    {  // random shuffle (use this instead of thrust::shuffle to use raft::random::RngState)
      rmm::device_uvector<float> random_numbers(mg_sample_buffer.size(), handle.get_stream());
      cugraph::detail::uniform_random_fill(handle.get_stream(),
                                           random_numbers.data(),
                                           random_numbers.size(),
                                           float{0.0},
                                           float{1.0},
                                           rng_state);
      thrust::sort_by_key(handle.get_thrust_policy(),
                          random_numbers.begin(),
                          random_numbers.end(),
                          mg_sample_buffer.begin());
    }

    if constexpr (multi_gpu) {
      auto const comm_rank = handle.get_comms().get_rank();
      auto const comm_size = handle.get_comms().get_size();

      std::vector<size_t> tx_value_counts(comm_size);
      for (int i = 0; i < comm_size; ++i) {
        tx_value_counts[i] = mg_sample_buffer.size() / comm_size;
      }

      std::vector<vertex_t> h_random_numbers;
      {
        rmm::device_uvector<vertex_t> d_random_numbers(mg_sample_buffer.size() % comm_size,
                                                       handle.get_stream());
        cugraph::detail::uniform_random_fill(handle.get_stream(),
                                             d_random_numbers.data(),
                                             d_random_numbers.size(),
                                             vertex_t{0},
                                             vertex_t{comm_size},
                                             rng_state);

        h_random_numbers.resize(d_random_numbers.size());

        raft::update_host(h_random_numbers.data(),
                          d_random_numbers.data(),
                          d_random_numbers.size(),
                          handle.get_stream());
      }

      for (int i = 0; i < static_cast<int>(mg_sample_buffer.size() % comm_size); i++) {
        tx_value_counts[h_random_numbers[i]]++;
      }

      std::tie(mg_sample_buffer, std::ignore) = cugraph::shuffle_values(
        handle.get_comms(), mg_sample_buffer.begin(), tx_value_counts, handle.get_stream());

      {  // random shuffle (use this instead of thrust::shuffle to use raft::random::RngState)
        rmm::device_uvector<float> random_numbers(mg_sample_buffer.size(), handle.get_stream());
        cugraph::detail::uniform_random_fill(handle.get_stream(),
                                             random_numbers.data(),
                                             random_numbers.size(),
                                             float{0.0},
                                             float{1.0},
                                             rng_state);
        thrust::sort_by_key(handle.get_thrust_policy(),
                            random_numbers.begin(),
                            random_numbers.end(),
                            mg_sample_buffer.begin());
      }

      auto buffer_sizes = cugraph::host_scalar_allgather(
        handle.get_comms(), mg_sample_buffer.size(), handle.get_stream());
      auto min_buffer_size = *std::min_element(buffer_sizes.begin(), buffer_sizes.end());
      if (min_buffer_size <= select_count / comm_size) {
        auto new_sizes    = std::vector<size_t>(comm_size, min_buffer_size);
        auto num_deficits = select_count - min_buffer_size * comm_size;
        for (int i = 0; i < comm_size; ++i) {
          auto delta = std::min(num_deficits, buffer_sizes[i] - min_buffer_size);
          new_sizes[i] += delta;
          num_deficits -= delta;
        }
        this_gpu_select_count = new_sizes[comm_rank];
      }
    }

    mg_sample_buffer.resize(this_gpu_select_count, handle.get_stream());
    mg_sample_buffer.shrink_to_fit(handle.get_stream());
  }

  if constexpr (multi_gpu) {
    mg_sample_buffer = cugraph::detail::shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
      handle, std::move(mg_sample_buffer), partition_range_lasts);
  }

  if (given_set) {
    auto const comm_rank = handle.get_comms().get_rank();
    thrust::gather(handle.get_thrust_policy(),
                   thrust::make_transform_iterator(
                     mg_sample_buffer.begin(),
                     cugraph::detail::shift_left_t<vertex_t>{
                       multi_gpu ? (comm_rank ? partition_range_lasts[comm_rank - 1] : 0) : 0}),
                   thrust::make_transform_iterator(
                     mg_sample_buffer.end(),
                     cugraph::detail::shift_left_t<vertex_t>{
                       multi_gpu ? (comm_rank ? partition_range_lasts[comm_rank - 1] : 0) : 0}),
                   (*given_set).begin(),
                   mg_sample_buffer.begin());
  }

  if (sort_vertices) {
    thrust::sort(handle.get_thrust_policy(), mg_sample_buffer.begin(), mg_sample_buffer.end());
  }

  return mg_sample_buffer;
}

}  // namespace cugraph
