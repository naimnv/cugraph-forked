/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "prims/fill_edge_property.cuh"
#include "prims/reduce_op.cuh"
#include "prims/transform_e.cuh"
#include "prims/transform_reduce_e_by_src_dst_key.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "utilities/collect_comm.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>

#include <raft/core/handle.hpp>

#include <thrust/fill.h>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
weight_t matching(raft::handle_t const& handle,
                  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                  edge_property_view_t<edge_t, weight_t const*> edge_weight_view,
                  raft::device_span<vertex_t> suitors)
{
  using graph_view_t = cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>;
  graph_view_t current_graph_view(graph_view);

  // edge mask
  cugraph::edge_property_t<graph_view_t, bool> edge_masks_even(handle, current_graph_view);
  cugraph::fill_edge_property(handle, current_graph_view, bool{false}, edge_masks_even);

  cugraph::edge_property_t<graph_view_t, bool> edge_masks_odd(handle, current_graph_view);
  cugraph::fill_edge_property(handle, current_graph_view, bool{false}, edge_masks_odd);

  cugraph::transform_e(
    handle,
    current_graph_view,
    cugraph::edge_src_dummy_property_t{}.view(),
    cugraph::edge_dst_dummy_property_t{}.view(),
    cugraph::edge_dummy_property_t{}.view(),
    [] __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) {
      return !(src == dst);  // mask out self-loop
    },
    edge_masks_even.mutable_view());

  current_graph_view.attach_edge_mask(edge_masks_even.view());

  // 2. initialize distances and predecessors

  auto constexpr invalid_distance = weight_t{-1.0};
  auto constexpr invalid_vertex   = invalid_vertex_id<vertex_t>::value;

  rmm::device_uvector<weight_t> offers_from_suitors(
    current_graph_view.local_vertex_partition_range_size(), handle.get_stream());

  thrust::fill(handle.get_thrust_policy(), suitors.begin(), suitors.end(), invalid_vertex);
  thrust::fill(handle.get_thrust_policy(),
               offers_from_suitors.begin(),
               offers_from_suitors.end(),
               weight_t{0.0});

  edge_src_property_t<graph_view_t, vertex_t> src_key_cache(handle);

  edge_dst_property_t<graph_view_t, vertex_t> dst_suitor_cache(handle);
  edge_dst_property_t<graph_view_t, weight_t> dst_offer_cache(handle);

  rmm::device_uvector<vertex_t> local_vertices(
    current_graph_view.local_vertex_partition_range_size(), handle.get_stream());

  detail::sequence_fill(handle.get_stream(),
                        local_vertices.begin(),
                        local_vertices.size(),
                        current_graph_view.local_vertex_partition_range_first());

  // auto vertex_begin =
  //   thrust::make_counting_iterator(current_graph_view.local_vertex_partition_range_first());
  // auto vertex_end =
  //   thrust::make_counting_iterator(current_graph_view.local_vertex_partition_range_last());

  // thrust::copy(handle.get_thrust_policy(), vertex_begin, vertex_end, local_vertices.begin());

  vertex_t loop_counter = 0;
  while (true) {
    if constexpr (graph_view_t::is_multi_gpu) {
      src_key_cache = edge_src_property_t<graph_view_t, vertex_t>(handle, current_graph_view);
      update_edge_src_property(handle, current_graph_view, local_vertices.begin(), src_key_cache);

      dst_suitor_cache = edge_dst_property_t<graph_view_t, vertex_t>(handle, current_graph_view);
      dst_offer_cache  = edge_dst_property_t<graph_view_t, weight_t>(handle, current_graph_view);

      update_edge_dst_property(handle, current_graph_view, suitors.begin(), dst_suitor_cache);
      update_edge_dst_property(
        handle, current_graph_view, offers_from_suitors.begin(), dst_offer_cache);
    }

    auto dst_input_property_values =
      graph_view_t::is_multi_gpu
        ? view_concat(dst_suitor_cache.view(), dst_offer_cache.view())
        : view_concat(detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(suitors.begin(),
                                                                                    vertex_t{0}),
                      detail::edge_minor_property_view_t<vertex_t, weight_t const*>(
                        offers_from_suitors.begin(), weight_t{0}));

    if (graph_view_t::is_multi_gpu) {
      auto const comm_rank = handle.get_comms().get_rank();
      auto const comm_size = handle.get_comms().get_size();

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto local_vertices_title = std::string("local_vertices_").append(std::to_string(comm_rank));

      raft::print_device_vector(
        local_vertices_title.c_str(), local_vertices.begin(), local_vertices.size(), std::cout);

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto offer_title = std::string("offers_").append(std::to_string(comm_rank));
      raft::print_device_vector(offer_title.c_str(),
                                offers_from_suitors.begin(),
                                current_graph_view.local_vertex_partition_range_size(),
                                std::cout);

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto partner_title = std::string("partners_").append(std::to_string(comm_rank));
      raft::print_device_vector(partner_title.c_str(),
                                suitors.begin(),
                                current_graph_view.local_vertex_partition_range_size(),
                                std::cout);
    }

    rmm::device_uvector<vertex_t> candidates(0, handle.get_stream());
    rmm::device_uvector<weight_t> offers_from_candidates(0, handle.get_stream());
    rmm::device_uvector<vertex_t> targets(0, handle.get_stream());
    rmm::device_uvector<vertex_t> curr_suitors_of_targets(0, handle.get_stream());

    std::forward_as_tuple(candidates,
                          std::tie(offers_from_candidates, targets, curr_suitors_of_targets)) =
      cugraph::transform_reduce_e_by_src_key(
        handle,
        current_graph_view,
        cugraph::edge_src_dummy_property_t{}.view(),
        dst_input_property_values,
        edge_weight_view,
        graph_view_t::is_multi_gpu
          ? src_key_cache.view()
          : detail::edge_major_property_view_t<vertex_t, vertex_t const*>(local_vertices.begin()),
        [] __device__(auto src,
                      auto dst,
                      thrust::nullopt_t,
                      thrust::tuple<vertex_t, weight_t> dst_suitor_offer,
                      auto wt) {
          auto suitor_of_dst     = thrust::get<0>(dst_suitor_offer);
          auto offer_from_suitor = thrust::get<1>(dst_suitor_offer);

          printf("src = %d dst = %d wt = %f suitor_of_dst = %d, offer_from_suitor = %f\n",
                 static_cast<int>(src),
                 static_cast<int>(dst),
                 static_cast<float>(wt),
                 static_cast<int>(suitor_of_dst),
                 static_cast<float>(offer_from_suitor));

          auto is_better_offer =
            (offer_from_suitor < wt) || ((offer_from_suitor == wt) && (suitor_of_dst < src));

          return is_better_offer
                   ? thrust::make_tuple(wt, dst, suitor_of_dst)
                   : thrust::make_tuple(invalid_distance, invalid_vertex, invalid_vertex);
        },
        thrust::make_tuple(invalid_distance, invalid_vertex, invalid_vertex),
        reduce_op::maximum<thrust::tuple<weight_t, vertex_t, vertex_t>>{},
        true);

    if (graph_view_t::is_multi_gpu) {
      auto const comm_rank = handle.get_comms().get_rank();
      auto const comm_size = handle.get_comms().get_size();

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto edge_reduced_src_keys_title =
        std::string("edge_reduced_src_keys_").append(std::to_string(comm_rank));
      raft::print_device_vector(
        edge_reduced_src_keys_title.c_str(), candidates.begin(), candidates.size(), std::cout);

      auto optimal_offers_title = std::string("from_cans_").append(std::to_string(comm_rank));
      raft::print_device_vector(optimal_offers_title.c_str(),
                                offers_from_candidates.begin(),
                                offers_from_candidates.size(),
                                std::cout);

      auto targets_title = std::string("targets_").append(std::to_string(comm_rank));
      raft::print_device_vector(targets_title.c_str(), targets.begin(), targets.size(), std::cout);

      auto cpt = std::string("cst_").append(std::to_string(comm_rank));
      raft::print_device_vector(
        cpt.c_str(), curr_suitors_of_targets.begin(), curr_suitors_of_targets.size(), std::cout);
    }

    if constexpr (graph_view_t::is_multi_gpu) {
      auto vertex_partition_range_lasts = current_graph_view.vertex_partition_range_lasts();

      rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(
        vertex_partition_range_lasts.size(), handle.get_stream());

      raft::update_device(d_vertex_partition_range_lasts.data(),
                          vertex_partition_range_lasts.data(),
                          vertex_partition_range_lasts.size(),
                          handle.get_stream());

      auto& major_comm = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      auto func = cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
        raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                          d_vertex_partition_range_lasts.size()),
        major_comm_size,
        minor_comm_size};

      rmm::device_uvector<size_t> d_tx_value_counts(0, handle.get_stream());

      auto triplet_first = thrust::make_zip_iterator(candidates.begin(),
                                                     offers_from_candidates.begin(),
                                                     targets.begin(),
                                                     curr_suitors_of_targets.begin());

      d_tx_value_counts = cugraph::groupby_and_count(
        triplet_first,
        triplet_first + candidates.size(),
        [func] __device__(auto val) { return func(thrust::get<2>(val)); },
        handle.get_comms().get_size(),
        std::numeric_limits<vertex_t>::max(),
        handle.get_stream());

      std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
      raft::update_host(h_tx_value_counts.data(),
                        d_tx_value_counts.data(),
                        d_tx_value_counts.size(),
                        handle.get_stream());
      handle.sync_stream();

      std::forward_as_tuple(
        std::tie(candidates, offers_from_candidates, targets, curr_suitors_of_targets),
        std::ignore) = shuffle_values(handle.get_comms(),
                                      thrust::make_zip_iterator(candidates.begin(),
                                                                offers_from_candidates.begin(),
                                                                targets.begin(),
                                                                curr_suitors_of_targets.begin()),
                                      h_tx_value_counts,
                                      handle.get_stream());
    }

    if (graph_view_t::is_multi_gpu) {
      auto const comm_rank = handle.get_comms().get_rank();
      auto const comm_size = handle.get_comms().get_size();

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto edge_reduced_src_keys_title =
        std::string("edge_reduced_src_keys_").append(std::to_string(comm_rank));
      raft::print_device_vector(
        edge_reduced_src_keys_title.c_str(), candidates.begin(), candidates.size(), std::cout);

      auto optimal_offers_title = std::string("from_cans_").append(std::to_string(comm_rank));
      raft::print_device_vector(optimal_offers_title.c_str(),
                                offers_from_candidates.begin(),
                                offers_from_candidates.size(),
                                std::cout);

      auto targets_title = std::string("targets_").append(std::to_string(comm_rank));
      raft::print_device_vector(targets_title.c_str(), targets.begin(), targets.size(), std::cout);

      auto cpt = std::string("cst_").append(std::to_string(comm_rank));
      raft::print_device_vector(
        cpt.c_str(), curr_suitors_of_targets.begin(), curr_suitors_of_targets.size(), std::cout);
    }

    using flag_t                        = uint8_t;
    rmm::device_uvector<flag_t> updated = rmm::device_uvector<flag_t>(
      current_graph_view.local_vertex_partition_range_size(), handle.get_stream());

    thrust::fill(handle.get_thrust_policy(), updated.begin(), updated.end(), flag_t{false});

    auto itr_to_tuples = thrust::make_zip_iterator(thrust::make_tuple(
      offers_from_candidates.begin(), candidates.begin(), curr_suitors_of_targets.begin()));

    thrust::sort_by_key(handle.get_thrust_policy(), targets.begin(), targets.end(), itr_to_tuples);

    auto nr_unique_targets = thrust::count_if(handle.get_thrust_policy(),
                                              thrust::make_counting_iterator(size_t{0}),
                                              thrust::make_counting_iterator(targets.size()),
                                              is_first_in_run_t<vertex_t const*>{targets.data()});

    rmm::device_uvector<vertex_t> unique_targets(nr_unique_targets, handle.get_stream());
    rmm::device_uvector<weight_t> best_offers_to_targets(nr_unique_targets, handle.get_stream());
    rmm::device_uvector<vertex_t> best_candidates(nr_unique_targets, handle.get_stream());
    rmm::device_uvector<vertex_t> curr_suitors_of_unique_targets(nr_unique_targets,
                                                                 handle.get_stream());

    auto itr_to_reduced_tuples =
      thrust::make_zip_iterator(thrust::make_tuple(best_offers_to_targets.begin(),
                                                   best_candidates.begin(),
                                                   curr_suitors_of_unique_targets.begin()));

    auto new_end = thrust::reduce_by_key(
      handle.get_thrust_policy(),
      targets.begin(),
      targets.end(),
      itr_to_tuples,
      unique_targets.begin(),
      itr_to_reduced_tuples,
      thrust::equal_to<vertex_t>{},
      [] __device__(auto pair1, auto pair2) {
        auto left  = thrust::make_tuple(thrust::get<0>(pair1), thrust::get<1>(pair1));
        auto right = thrust::make_tuple(thrust::get<0>(pair2), thrust::get<1>(pair2));
        return (left > right) ? pair1 : pair2;
      });

    vertex_t nr_reduces_tuples =
      static_cast<vertex_t>(thrust::distance(unique_targets.begin(), new_end.first));

    targets                 = std::move(unique_targets);
    offers_from_candidates  = std::move(best_offers_to_targets);
    candidates              = std::move(best_candidates);
    curr_suitors_of_targets = std::move(curr_suitors_of_unique_targets);

    kv_store_t<vertex_t, vertex_t, false> target_src_map(targets.begin(),
                                                         targets.end(),
                                                         candidates.begin(),
                                                         invalid_vertex_id<vertex_t>::value,
                                                         invalid_vertex_id<vertex_t>::value,
                                                         handle.get_stream());

    rmm::device_uvector<vertex_t> match_of_srcs(0, handle.get_stream());

    if (graph_view_t::is_multi_gpu) {
      auto& comm           = handle.get_comms();
      auto const comm_size = comm.get_size();
      auto& major_comm     = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      auto partitions_range_lasts = graph_view.vertex_partition_range_lasts();
      rmm::device_uvector<vertex_t> d_partitions_range_lasts(partitions_range_lasts.size(),
                                                             handle.get_stream());

      raft::update_device(d_partitions_range_lasts.data(),
                          partitions_range_lasts.data(),
                          partitions_range_lasts.size(),
                          handle.get_stream());

      cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t> vertex_to_gpu_id_op{
        raft::device_span<vertex_t const>(d_partitions_range_lasts.data(),
                                          d_partitions_range_lasts.size()),
        major_comm_size,
        minor_comm_size};

      // cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t> vertex_to_gpu_id_op{
      //   comm_size, major_comm_size, minor_comm_size};

      match_of_srcs = cugraph::collect_values_for_keys(
        handle, target_src_map.view(), candidates.begin(), candidates.end(), vertex_to_gpu_id_op);
    } else {
      match_of_srcs.resize(candidates.size(), handle.get_stream());

      target_src_map.view().find(
        candidates.begin(), candidates.end(), match_of_srcs.begin(), handle.get_stream());
    }

    if (graph_view_t::is_multi_gpu) {
      auto const comm_rank = handle.get_comms().get_rank();
      auto const comm_size = handle.get_comms().get_size();

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto edge_reduced_src_keys_title = std::string("srcs_").append(std::to_string(comm_rank));
      raft::print_device_vector(
        edge_reduced_src_keys_title.c_str(), candidates.begin(), candidates.size(), std::cout);
      auto targets_title = std::string("targets_").append(std::to_string(comm_rank));
      raft::print_device_vector(targets_title.c_str(), targets.begin(), targets.size(), std::cout);

      auto mt_title = std::string("m_srcs_").append(std::to_string(comm_rank));
      raft::print_device_vector(
        mt_title.c_str(), match_of_srcs.begin(), match_of_srcs.size(), std::cout);

      auto optimal_offers_title = std::string("from_cans_").append(std::to_string(comm_rank));
      raft::print_device_vector(optimal_offers_title.c_str(),
                                offers_from_candidates.begin(),
                                offers_from_candidates.size(),
                                std::cout);

      auto cpt = std::string("cst_").append(std::to_string(comm_rank));
      raft::print_device_vector(
        cpt.c_str(), curr_suitors_of_targets.begin(), curr_suitors_of_targets.size(), std::cout);
    }

    if (graph_view_t::is_multi_gpu) {
      auto const comm_rank = handle.get_comms().get_rank();
      auto const comm_size = handle.get_comms().get_size();
      auto partners_title  = std::string("partners_").append(std::to_string(comm_rank));
      raft::print_device_vector(partners_title.c_str(), suitors.begin(), suitors.size(), std::cout);
    }

    // match of srcs are same as targets then mask out its edges

    using flag_t                                  = uint8_t;
    rmm::device_uvector<flag_t> is_vertex_matched = rmm::device_uvector<flag_t>(
      current_graph_view.local_vertex_partition_range_size(), handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 is_vertex_matched.begin(),
                 is_vertex_matched.end(),
                 flag_t{false});

    cugraph::edge_src_property_t<graph_view_t, flag_t> src_match_flags(handle);
    cugraph::edge_dst_property_t<graph_view_t, flag_t> dst_match_flags(handle);

    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(thrust::make_tuple(match_of_srcs.begin(),
                                                   targets.begin(),
                                                   candidates.begin(),
                                                   offers_from_candidates.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(
        match_of_srcs.end(), targets.end(), candidates.end(), offers_from_candidates.end())),
      [suitors             = suitors.begin(),
       offers_from_suitors = offers_from_suitors.begin(),
       is_vertex_matched   = is_vertex_matched.data(),
       v_first =
         current_graph_view.local_vertex_partition_range_first()] __device__(auto msrc_tgt) {
        auto msrc        = thrust::get<0>(msrc_tgt);
        auto tgt         = thrust::get<1>(msrc_tgt);
        auto src         = thrust::get<2>(msrc_tgt);
        auto offer_value = thrust::get<3>(msrc_tgt);

        if (msrc != invalid_vertex && msrc == tgt) {
          printf("===> %d found  %d as match with value %f \n",
                 static_cast<int>(tgt),
                 static_cast<int>(src),
                 static_cast<float>(offer_value));
          auto tgt_offset                 = tgt - v_first;
          is_vertex_matched[tgt_offset]   = flag_t{true};
          suitors[tgt_offset]             = src;
          offers_from_suitors[tgt_offset] = offer_value;
        }
      });

    if (current_graph_view.compute_number_of_edges(handle) == 0) { break; }

    if constexpr (graph_view_t::is_multi_gpu) {
      src_match_flags =
        cugraph::edge_src_property_t<graph_view_t, flag_t>(handle, current_graph_view);
      dst_match_flags =
        cugraph::edge_dst_property_t<graph_view_t, flag_t>(handle, current_graph_view);

      cugraph::update_edge_src_property(
        handle, current_graph_view, is_vertex_matched.begin(), src_match_flags);

      cugraph::update_edge_dst_property(
        handle, current_graph_view, is_vertex_matched.begin(), dst_match_flags);
    }

    if (loop_counter % 2 == 0) {
      cugraph::transform_e(
        handle,
        current_graph_view,
        graph_view_t::is_multi_gpu
          ? src_match_flags.view()
          : detail::edge_major_property_view_t<vertex_t, flag_t const*>(is_vertex_matched.begin()),
        graph_view_t::is_multi_gpu ? dst_match_flags.view()
                                   : detail::edge_minor_property_view_t<vertex_t, flag_t const*>(
                                       is_vertex_matched.begin(), vertex_t{0}),
        cugraph::edge_dummy_property_t{}.view(),
        [loop_counter] __device__(
          auto src, auto dst, auto is_src_matched, auto is_dst_matched, thrust::nullopt_t) {
          return !((is_src_matched == uint8_t{true}) || (is_dst_matched == uint8_t{true}));
        },
        edge_masks_odd.mutable_view());

      if (current_graph_view.has_edge_mask()) current_graph_view.clear_edge_mask();
      cugraph::fill_edge_property(handle, current_graph_view, bool{false}, edge_masks_even);
      current_graph_view.attach_edge_mask(edge_masks_odd.view());
    } else {
      cugraph::transform_e(
        handle,
        current_graph_view,
        src_match_flags.view(),
        dst_match_flags.view(),
        cugraph::edge_dummy_property_t{}.view(),
        [loop_counter] __device__(
          auto src, auto dst, auto is_src_matched, auto is_dst_matched, thrust::nullopt_t) {
          return !((is_src_matched == uint8_t{true}) || (is_dst_matched == uint8_t{true}));
        },
        edge_masks_even.mutable_view());

      if (current_graph_view.has_edge_mask()) current_graph_view.clear_edge_mask();
      cugraph::fill_edge_property(handle, current_graph_view, bool{false}, edge_masks_odd);
      current_graph_view.attach_edge_mask(edge_masks_even.view());
    }

    loop_counter++;
  }

  weight_t sum_matched_edge_weights = thrust::reduce(
    handle.get_thrust_policy(), offers_from_suitors.begin(), offers_from_suitors.end());

  if constexpr (graph_view_t::is_multi_gpu) {
    sum_matched_edge_weights = host_scalar_allreduce(
      handle.get_comms(), sum_matched_edge_weights, raft::comms::op_t::SUM, handle.get_stream());
  }

  return sum_matched_edge_weights;
}
}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
weight_t matching(raft::handle_t const& handle,
                  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                  edge_property_view_t<edge_t, weight_t const*> edge_weight_view,
                  raft::device_span<vertex_t> suitors)
{
  return detail::matching(handle, graph_view, edge_weight_view, suitors);
}

}  // namespace cugraph