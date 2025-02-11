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

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/visitors/generic_cascaded_dispatch.hpp>
#include <cugraph_c/graph.h>

#include <c_api/abstract_functor.hpp>
#include <c_api/array.hpp>
#include <c_api/error.hpp>
#include <c_api/graph.hpp>
#include <c_api/resource_handle.hpp>

#include <limits>

namespace {

struct create_graph_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph_graph_properties_t const* properties_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* src_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* dst_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* weights_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* edge_ids_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* edge_types_;
  bool_t renumber_;
  bool_t check_;
  data_type_id_t edge_type_;
  cugraph::c_api::cugraph_graph_t* result_{};

  create_graph_functor(raft::handle_t const& handle,
                       cugraph_graph_properties_t const* properties,
                       cugraph::c_api::cugraph_type_erased_device_array_view_t const* src,
                       cugraph::c_api::cugraph_type_erased_device_array_view_t const* dst,
                       cugraph::c_api::cugraph_type_erased_device_array_view_t const* weights,
                       cugraph::c_api::cugraph_type_erased_device_array_view_t const* edge_ids,
                       cugraph::c_api::cugraph_type_erased_device_array_view_t const* edge_types,
                       bool_t renumber,
                       bool_t check,
                       data_type_id_t edge_type)
    : abstract_functor(),
      properties_(properties),
      handle_(handle),
      src_(src),
      dst_(dst),
      weights_(weights),
      edge_ids_(edge_ids),
      edge_types_(edge_types),
      renumber_(renumber),
      check_(check),
      edge_type_(edge_type)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename edge_type_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    if constexpr (!multi_gpu || !cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      // using edge_type_t = int32_t;

      std::optional<rmm::device_uvector<vertex_t>> new_number_map;

      std::optional<cugraph::edge_property_t<
        cugraph::graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
        thrust::tuple<edge_t, edge_type_t>>>
        new_edge_properties;

      rmm::device_uvector<vertex_t> edgelist_srcs(src_->size_, handle_.get_stream());
      rmm::device_uvector<vertex_t> edgelist_dsts(dst_->size_, handle_.get_stream());

      raft::copy<vertex_t>(
        edgelist_srcs.data(), src_->as_type<vertex_t>(), src_->size_, handle_.get_stream());
      raft::copy<vertex_t>(
        edgelist_dsts.data(), dst_->as_type<vertex_t>(), dst_->size_, handle_.get_stream());

      std::optional<rmm::device_uvector<weight_t>> edgelist_weights =
        weights_
          ? std::make_optional(rmm::device_uvector<weight_t>(weights_->size_, handle_.get_stream()))
          : std::nullopt;

      if (edgelist_weights) {
        raft::copy<weight_t>(edgelist_weights->data(),
                             weights_->as_type<weight_t>(),
                             weights_->size_,
                             handle_.get_stream());
      }

      std::optional<std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_t>>>
        edgelist_edge_tuple{};

      if (edge_types_ && edge_ids_) {
        auto edgelist_edge_types =
          rmm::device_uvector<edge_type_t>(edge_types_->size_, handle_.get_stream());

        raft::copy<edge_type_t>(edgelist_edge_types.data(),
                                edge_types_->as_type<edge_type_t>(),
                                edge_types_->size_,
                                handle_.get_stream());

        auto edgelist_edge_ids =
          rmm::device_uvector<edge_t>(edge_ids_->size_, handle_.get_stream());

        raft::copy<edge_t>(edgelist_edge_ids.data(),
                           edge_types_->as_type<edge_t>(),
                           edge_types_->size_,
                           handle_.get_stream());

        edgelist_edge_tuple =
          std::make_tuple(std::move(edgelist_edge_ids), std::move(edgelist_edge_types));
      }

      // Here's the error.  If store_transposed is true then this needs to be flipped...
      std::tie(store_transposed ? edgelist_dsts : edgelist_srcs,
               store_transposed ? edgelist_srcs : edgelist_dsts,
               edgelist_weights) =
        cugraph::detail::shuffle_edgelist_by_gpu_id<vertex_t, weight_t>(
          handle_,
          std::move(store_transposed ? edgelist_dsts : edgelist_srcs),
          std::move(store_transposed ? edgelist_srcs : edgelist_dsts),
          std::move(edgelist_weights));

      auto graph =
        new cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(handle_);

      rmm::device_uvector<vertex_t>* number_map =
        new rmm::device_uvector<vertex_t>(0, handle_.get_stream());

      auto edge_properties = new cugraph::edge_property_t<
        cugraph::graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
        thrust::tuple<edge_t, edge_type_t>>(handle_);

      std::tie(*graph, new_edge_properties, new_number_map) =
        cugraph::create_graph_from_edgelist<vertex_t,
                                            edge_t,
                                            weight_t,
                                            edge_type_t,
                                            store_transposed,
                                            multi_gpu>(
          handle_,
          std::nullopt,
          std::move(edgelist_srcs),
          std::move(edgelist_dsts),
          std::move(edgelist_weights),
          std::move(edgelist_edge_tuple),
          cugraph::graph_properties_t{properties_->is_symmetric, properties_->is_multigraph},
          renumber_,
          check_);

      if (renumber_) {
        *number_map = std::move(new_number_map.value());
      } else {
        number_map->resize(graph->number_of_vertices(), handle_.get_stream());
        cugraph::detail::sequence_fill(handle_.get_stream(),
                                       number_map->data(),
                                       number_map->size(),
                                       graph->view().local_vertex_partition_range_first());
      }

      if (new_edge_properties) { *edge_properties = std::move(new_edge_properties.value()); }

      // Set up return
      auto result = new cugraph::c_api::cugraph_graph_t{
        src_->type_,
        edge_type_,
        weights_ ? weights_->type_ : data_type_id_t::FLOAT32,
        edge_types_ ? edge_types_->type_ : data_type_id_t::INT32,
        store_transposed,
        multi_gpu,
        graph,
        number_map,
        edge_properties};

      result_ = reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(result);
    }
  }
};

struct destroy_graph_functor : public cugraph::c_api::abstract_functor {
  void* graph_;
  void* number_map_;
  void* edge_properties_;

  destroy_graph_functor(void* graph, void* number_map, void* edge_properties)
    : abstract_functor(), graph_(graph), number_map_(number_map), edge_properties_(edge_properties)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename edge_type_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    auto internal_graph_pointer =
      reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>*>(
        graph_);

    delete internal_graph_pointer;

    auto internal_number_map_pointer =
      reinterpret_cast<rmm::device_uvector<vertex_t>*>(number_map_);

    delete internal_number_map_pointer;

    auto internal_edge_property_pointer = reinterpret_cast<cugraph::edge_property_t<
      cugraph::graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
      thrust::tuple<edge_t, edge_type_t>>*>(edge_properties_);

    delete internal_edge_property_pointer;
  }
};

}  // namespace

extern "C" cugraph_error_code_t cugraph_mg_graph_create(
  const cugraph_resource_handle_t* handle,
  const cugraph_graph_properties_t* properties,
  const cugraph_type_erased_device_array_view_t* src,
  const cugraph_type_erased_device_array_view_t* dst,
  const cugraph_type_erased_device_array_view_t* weights,
  const cugraph_type_erased_device_array_view_t* edge_ids,
  const cugraph_type_erased_device_array_view_t* edge_types,
  bool_t store_transposed,
  size_t num_edges,
  bool_t check,
  cugraph_graph_t** graph,
  cugraph_error_t** error)
{
  constexpr bool multi_gpu = true;
  constexpr size_t int32_threshold{std::numeric_limits<int32_t>::max()};

  *graph = nullptr;
  *error = nullptr;

  auto p_handle = reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle);
  auto p_src =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(src);
  auto p_dst =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(dst);
  auto p_weights =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(weights);
  auto p_edge_ids =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(edge_ids);
  auto p_edge_types =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(edge_types);

  CAPI_EXPECTS(p_src->size_ == p_dst->size_,
               CUGRAPH_INVALID_INPUT,
               "Invalid input arguments: src size != dst size.",
               *error);
  CAPI_EXPECTS(p_src->type_ == p_dst->type_,
               CUGRAPH_INVALID_INPUT,
               "Invalid input arguments: src type != dst type.",
               *error);

  CAPI_EXPECTS((weights == nullptr) || (p_weights->size_ == p_src->size_),
               CUGRAPH_INVALID_INPUT,
               "Invalid input arguments: src size != weights size.",
               *error);

  data_type_id_t edge_type;
  data_type_id_t weight_type;

  if (num_edges < int32_threshold) {
    edge_type = p_src->type_;
  } else {
    edge_type = data_type_id_t::INT64;
  }

  if (weights != nullptr) {
    weight_type = p_weights->type_;
  } else {
    weight_type = data_type_id_t::FLOAT32;
  }

  CAPI_EXPECTS(
    (edge_types == nullptr && edge_ids == nullptr) ||
      (edge_types != nullptr && edge_ids != nullptr),
    CUGRAPH_INVALID_INPUT,
    "Invalid input arguments: either none or both of edge ids and edge types must be provided.",
    *error);

  CAPI_EXPECTS((edge_types == nullptr && edge_ids == nullptr) || (p_edge_ids->type_ == edge_type),
               CUGRAPH_INVALID_INPUT,
               "Invalid input arguments: Edge id type must match edge type",
               *error);

  CAPI_EXPECTS((edge_types == nullptr && edge_ids == nullptr) ||
                 (p_edge_ids->size_ == p_src->size_ && p_edge_types->size_ == p_dst->size_),
               CUGRAPH_INVALID_INPUT,
               "Invalid input arguments: src size != edge prop size",
               *error);

  data_type_id_t edge_type_type;
  if (edge_types == nullptr) {
    edge_type_type = data_type_id_t::INT32;
  } else {
    edge_type_type = p_edge_types->type_;
  }

  create_graph_functor functor(*p_handle->handle_,
                               properties,
                               p_src,
                               p_dst,
                               p_weights,
                               p_edge_ids,
                               p_edge_types,
                               bool_t::TRUE,
                               check,
                               edge_type);

  try {
    cugraph::dispatch::vertex_dispatcher(cugraph::c_api::dtypes_mapping[p_src->type_],
                                         cugraph::c_api::dtypes_mapping[edge_type],
                                         cugraph::c_api::dtypes_mapping[weight_type],
                                         cugraph::c_api::dtypes_mapping[edge_type_type],
                                         store_transposed,
                                         multi_gpu,
                                         functor);

    if (functor.error_code_ != CUGRAPH_SUCCESS) {
      *error = reinterpret_cast<cugraph_error_t*>(functor.error_.release());
      return functor.error_code_;
    }

    *graph = reinterpret_cast<cugraph_graph_t*>(functor.result_);
  } catch (std::exception const& ex) {
    *error = reinterpret_cast<cugraph_error_t*>(new cugraph::c_api::cugraph_error_t{ex.what()});
    return CUGRAPH_UNKNOWN_ERROR;
  }

  return CUGRAPH_SUCCESS;
}

extern "C" void cugraph_mg_graph_free(cugraph_graph_t* ptr_graph)
{
  if (ptr_graph != NULL) {
    auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(ptr_graph);

    destroy_graph_functor functor(
      internal_pointer->graph_, internal_pointer->number_map_, internal_pointer->edge_properties_);

    cugraph::dispatch::vertex_dispatcher(
      cugraph::c_api::dtypes_mapping[internal_pointer->vertex_type_],
      cugraph::c_api::dtypes_mapping[internal_pointer->edge_type_],
      cugraph::c_api::dtypes_mapping[internal_pointer->weight_type_],
      cugraph::c_api::dtypes_mapping[internal_pointer->edge_type_type_],
      internal_pointer->store_transposed_,
      internal_pointer->multi_gpu_,
      functor);

    delete internal_pointer;
  }
}
