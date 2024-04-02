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

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>

#include <raft/core/handle.hpp>

#include <thrust/fill.h>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void matching(raft::handle_t const& handle,
              cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
              edge_property_view_t<edge_t, weight_t const*> edge_weight_view,
              vertex_t* suitors)
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

  // rmm::device_uvector<size_t> sort_indices(edge_srcs.size(), handle.get_stream());
  // thrust::tabulate(
  //   handle.get_thrust_policy(),
  //   sort_indices.begin(),
  //   sort_indices.end(),
  //   [offset_lasts   = raft::device_span<size_t const>(offsets.begin() + 1, offsets.end()),
  //    source_indices = raft::device_span<size_t const>(source_indices.data(),
  //                                                     source_indices.size())] __device__(size_t
  //                                                     i) {
  //     auto idx = static_cast<size_t>(thrust::distance(
  //       offset_lasts.begin(),
  //       thrust::upper_bound(thrust::seq, offset_lasts.begin(), offset_lasts.end(), i)));
  //     return source_indices[idx];
  //   });

  raft::device_span<vertex_t> partners(suitors,
                                       current_graph_view.local_vertex_partition_range_size());
  rmm::device_uvector<weight_t> offers(current_graph_view.local_vertex_partition_range_size(),
                                       handle.get_stream());

  thrust::fill(handle.get_thrust_policy(), partners.begin(), partners.end(), invalid_vertex);

  thrust::fill(handle.get_thrust_policy(), offers.begin(), offers.end(), invalid_vertex);

  // edge_src_property_t<graph_view_t, vertex_t> src_partner_cache(handle);
  // edge_src_property_t<graph_view_t, weight_t> src_offer_cache(handle);
  edge_src_property_t<graph_view_t, vertex_t> src_key_cache(handle);

  edge_dst_property_t<graph_view_t, vertex_t> dst_partner_cache(handle);
  edge_dst_property_t<graph_view_t, weight_t> dst_offer_cache(handle);
  // edge_dst_property_t<graph_view_t, vertex_t> dst_key_cache(handle);

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

  vertex_t itr_cnt           = 0;
  int nr_of_updated_vertices = 0;
  while (true) {
    if constexpr (graph_view_t::is_multi_gpu) {
      //     src_partner_cache =
      //       edge_src_property_t<graph_view_t, vertex_t>(handle, current_graph_view);
      //     src_offer_cache = edge_src_property_t<graph_view_t, weight_t>(handle,
      // current_graph_view);

      src_key_cache = edge_src_property_t<graph_view_t, vertex_t>(handle, current_graph_view);

      dst_partner_cache = edge_dst_property_t<graph_view_t, vertex_t>(handle, current_graph_view);
      dst_offer_cache   = edge_dst_property_t<graph_view_t, weight_t>(handle, current_graph_view);

      // dst_key_cache = edge_dst_property_t<graph_view_t, vertex_t>(handle, current_graph_view);

      // update_edge_src_property(handle, current_graph_view, partners.begin(), src_partner_cache);
      // update_edge_src_property(handle, current_graph_view, offers.begin(), src_offer_cache);
      update_edge_src_property(handle, current_graph_view, local_vertices.begin(), src_key_cache);

      update_edge_dst_property(handle, current_graph_view, partners.begin(), dst_partner_cache);
      update_edge_dst_property(handle, current_graph_view, offers.begin(), dst_offer_cache);

      // update_edge_dst_property(handle, current_graph_view, local_vertices.begin(),
      // dst_key_cache);
    }

    // auto src_input_property_values =
    //   graph_view_t::is_multi_gpu
    //     ? view_concat(src_partner_cache.view(), src_offer_cache.view())
    //     : view_concat(detail::edge_major_property_view_t<vertex_t, vertex_t
    //     const*>(predecessors),
    //                   detail::edge_major_property_view_t<vertex_t, weight_t const*>(distances));

    auto dst_input_property_values =
      graph_view_t::is_multi_gpu
        ? view_concat(dst_partner_cache.view(), dst_offer_cache.view())
        : view_concat(detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                        partners.begin(), vertex_t{0}),
                      detail::edge_minor_property_view_t<vertex_t, weight_t const*>(offers.begin(),
                                                                                    weight_t{0}));

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
                                offers.begin(),
                                current_graph_view.local_vertex_partition_range_size(),
                                std::cout);

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto partner_title = std::string("partners_").append(std::to_string(comm_rank));
      raft::print_device_vector(partner_title.c_str(),
                                partners.begin(),
                                current_graph_view.local_vertex_partition_range_size(),
                                std::cout);
    }

    rmm::device_uvector<vertex_t> edge_reduced_src_keys(0, handle.get_stream());
    rmm::device_uvector<weight_t> optimal_offers(0, handle.get_stream());
    rmm::device_uvector<vertex_t> targets(0, handle.get_stream());
    rmm::device_uvector<vertex_t> cp_targets(0, handle.get_stream());

    std::forward_as_tuple(edge_reduced_src_keys, std::tie(optimal_offers, targets, cp_targets)) =
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
                      thrust::tuple<vertex_t, weight_t> dst_partner_offer,
                      auto wt) {
          auto dst_partner = thrust::get<0>(dst_partner_offer);
          auto dst_offer   = thrust::get<1>(dst_partner_offer);

          printf("src = %d dst = %d wt = %f dst_partner = %d, dst_offer = %f\n",
                 static_cast<int>(src),
                 static_cast<int>(dst),
                 static_cast<float>(wt),
                 static_cast<int>(dst_partner),
                 static_cast<float>(dst_offer));

          auto is_better_offer = (dst_offer < wt) || ((dst_offer == wt) && (dst_partner < src));

          return is_better_offer
                   ? thrust::make_tuple(wt, dst, dst_partner)
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
      raft::print_device_vector(edge_reduced_src_keys_title.c_str(),
                                edge_reduced_src_keys.begin(),
                                edge_reduced_src_keys.size(),
                                std::cout);

      auto optimal_offers_title = std::string("optimal_offers_").append(std::to_string(comm_rank));
      raft::print_device_vector(
        optimal_offers_title.c_str(), optimal_offers.begin(), optimal_offers.size(), std::cout);

      auto targets_title = std::string("targets_").append(std::to_string(comm_rank));
      raft::print_device_vector(targets_title.c_str(), targets.begin(), targets.size(), std::cout);

      auto cpt = std::string("cpt_").append(std::to_string(comm_rank));
      raft::print_device_vector(cpt.c_str(), cp_targets.begin(), cp_targets.size(), std::cout);
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

      auto triplet_first = thrust::make_zip_iterator(
        edge_reduced_src_keys.begin(), optimal_offers.begin(), targets.begin(), cp_targets.begin());

      d_tx_value_counts = cugraph::groupby_and_count(
        triplet_first,
        triplet_first + edge_reduced_src_keys.size(),
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

      std::forward_as_tuple(std::tie(edge_reduced_src_keys, optimal_offers, targets, cp_targets),
                            std::ignore) =
        shuffle_values(handle.get_comms(),
                       thrust::make_zip_iterator(edge_reduced_src_keys.begin(),
                                                 optimal_offers.begin(),
                                                 targets.begin(),
                                                 cp_targets.begin()),
                       h_tx_value_counts,
                       handle.get_stream());
    }

    if (graph_view_t::is_multi_gpu) {
      auto const comm_rank = handle.get_comms().get_rank();
      auto const comm_size = handle.get_comms().get_size();

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto edge_reduced_src_keys_title =
        std::string("edge_reduced_src_keys_").append(std::to_string(comm_rank));
      raft::print_device_vector(edge_reduced_src_keys_title.c_str(),
                                edge_reduced_src_keys.begin(),
                                edge_reduced_src_keys.size(),
                                std::cout);

      auto optimal_offers_title = std::string("optimal_offers_").append(std::to_string(comm_rank));
      raft::print_device_vector(
        optimal_offers_title.c_str(), optimal_offers.begin(), optimal_offers.size(), std::cout);

      auto targets_title = std::string("targets_").append(std::to_string(comm_rank));
      raft::print_device_vector(targets_title.c_str(), targets.begin(), targets.size(), std::cout);

      auto cpt = std::string("cpt_").append(std::to_string(comm_rank));
      raft::print_device_vector(cpt.c_str(), cp_targets.begin(), cp_targets.size(), std::cout);
    }

    using flag_t                        = uint8_t;
    rmm::device_uvector<flag_t> updated = rmm::device_uvector<flag_t>(
      current_graph_view.local_vertex_partition_range_size(), handle.get_stream());

    thrust::fill(handle.get_thrust_policy(), updated.begin(), updated.end(), flag_t{false});

    auto values_itr = thrust::make_zip_iterator(thrust::make_tuple(
      optimal_offers.begin(), edge_reduced_src_keys.begin(), cp_targets.begin()));

    thrust::sort_by_key(handle.get_thrust_policy(), targets.begin(), targets.end(), values_itr);

    auto nr_unique = thrust::count_if(handle.get_thrust_policy(),
                                      thrust::make_counting_iterator(size_t{0}),
                                      thrust::make_counting_iterator(targets.size()),
                                      is_first_in_run_t<vertex_t const*>{targets.data()});

    rmm::device_uvector<vertex_t> r_targets(nr_unique, handle.get_stream());
    rmm::device_uvector<weight_t> r_offers(nr_unique, handle.get_stream());
    rmm::device_uvector<vertex_t> r_src_keys(nr_unique, handle.get_stream());
    rmm::device_uvector<vertex_t> r_cp_targets(nr_unique, handle.get_stream());

    auto r_values_itr =
      thrust::make_zip_iterator(r_offers.begin(), r_src_keys.begin(), r_cp_targets.begin());

    auto new_end = thrust::reduce_by_key(
      handle.get_thrust_policy(),
      targets.begin(),
      targets.end(),
      values_itr,
      r_targets.begin(),
      r_values_itr,
      thrust::equal_to<vertex_t>{},
      [] __device__(auto pair1, auto pair2) {
        auto left  = thrust::make_tuple(thrust::get<0>(pair1), thrust::get<1>(pair1));
        auto right = thrust::make_tuple(thrust::get<0>(pair2), thrust::get<1>(pair2));
        return (left > right) ? pair1 : pair2;
      });

    vertex_t nr_reduces_tuples =
      static_cast<vertex_t>(thrust::distance(r_targets.begin(), new_end.first));

    targets               = std::move(r_targets);
    optimal_offers        = std::move(r_offers);
    edge_reduced_src_keys = std::move(r_src_keys);
    cp_targets            = std::move(r_cp_targets);

    if (graph_view_t::is_multi_gpu) {
      auto const comm_rank = handle.get_comms().get_rank();
      auto const comm_size = handle.get_comms().get_size();

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto edge_reduced_src_keys_title = std::string("srcs_").append(std::to_string(comm_rank));
      raft::print_device_vector(edge_reduced_src_keys_title.c_str(),
                                edge_reduced_src_keys.begin(),
                                edge_reduced_src_keys.size(),
                                std::cout);
      auto targets_title = std::string("targets_").append(std::to_string(comm_rank));
      raft::print_device_vector(targets_title.c_str(), targets.begin(), targets.size(), std::cout);

      auto optimal_offers_title = std::string("optimal_offers_").append(std::to_string(comm_rank));
      raft::print_device_vector(
        optimal_offers_title.c_str(), optimal_offers.begin(), optimal_offers.size(), std::cout);

      auto cpt = std::string("cpt_").append(std::to_string(comm_rank));
      raft::print_device_vector(cpt.c_str(), cp_targets.begin(), cp_targets.size(), std::cout);
    }

    if (graph_view_t::is_multi_gpu) {
      auto const comm_rank = handle.get_comms().get_rank();
      auto const comm_size = handle.get_comms().get_size();
      auto partners_title  = std::string("partners_").append(std::to_string(comm_rank));
      raft::print_device_vector(
        partners_title.c_str(), partners.begin(), partners.size(), std::cout);
    }
    break;

    /*
    nr_of_updated_vertices =
      thrust::count(handle.get_thrust_policy(), updated.begin(), updated.end(), flag_t{true});

    if constexpr (graph_view_t::is_multi_gpu) {
      nr_of_updated_vertices = host_scalar_allreduce(
        handle.get_comms(), nr_of_updated_vertices, raft::comms::op_t::SUM, handle.get_stream());
    }

    itr_cnt++;
    std::cout << "itr_cnt: " << itr_cnt << std::endl;

    if (nr_of_updated_vertices == 0) {
      std::cout << "No more updates\n";
      break;
    }
    */
  }
}
}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void matching(raft::handle_t const& handle,
              graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
              edge_property_view_t<edge_t, weight_t const*> edge_weight_view,
              vertex_t* suitors)
{
  detail::matching(handle, graph_view, edge_weight_view, suitors);
}

}  // namespace cugraph