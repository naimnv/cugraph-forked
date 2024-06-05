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
#include "detail/graph_partition_utils.cuh"
#include "prims/extract_transform_e.cuh"
#include "prims/kv_store.cuh"
#include "utilities/collect_comm.cuh"

#include <cugraph/detail/collect_comm_wrapper.hpp>
#include <cugraph/detail/decompress_edge_partition.cuh>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/src_dst_lookup_container.hpp>
#include <cugraph/utilities/mask_utils.cuh>
#include <cugraph/utilities/misc_utils.cuh>

#include <raft/core/handle.hpp>

namespace cugraph {

namespace detail {
template <typename edge_t, typename edge_type_t>
struct compute_key_from_edge_id_and_type_t {
  int type_width{0};
  __host__ __device__ int operator()(edge_t id, edge_type_t type) const
  {
    return (((uint64_t)id) << type_width) | (uint64_t)(type & (~(~0 << type_width)));
  }
};
}  // namespace detail

template <typename edge_id_t, typename edge_type_t, typename vertex_t, typename value_t>
template <typename _edge_id_t, typename _edge_type_t, typename _vertex_t, typename _value_t>
struct lookup_container_t<edge_id_t, edge_type_t, vertex_t, value_t>::lookup_container_impl {
  static_assert(std::is_integral_v<edge_id_t>);
  static_assert(std::is_integral_v<edge_type_t>);

  static_assert(std::is_same_v<edge_type_t, _edge_type_t>);
  static_assert(std::is_same_v<edge_id_t, _edge_id_t>);
  static_assert(std::is_same_v<value_t, _value_t>);

  ~lookup_container_impl() { delete key_to_src_dst_kv_sotre; }
  lookup_container_impl() {}
  lookup_container_impl(raft::handle_t const& handle, size_t capacity)
  {
    auto invalid_vertex_id = cugraph::invalid_vertex_id<edge_id_t>::value;
    auto invalid_value = thrust::tuple<vertex_t, vertex_t>(invalid_vertex_id, invalid_vertex_id);

    key_to_src_dst_kv_sotre =
      new container_t(capacity, invalid_vertex_id, invalid_value, handle.get_stream());
  }

  void insert(raft::handle_t const& handle,
              rmm::device_uvector<uint64_t>& keys_to_insert,
              dataframe_buffer_type_t<value_t>&& values_to_insert)
  {
    key_to_src_dst_kv_sotre->insert(keys_to_insert.begin(),
                                    keys_to_insert.end(),
                                    cugraph::get_dataframe_buffer_begin(values_to_insert),
                                    handle.get_stream());
  }

  dataframe_buffer_type_t<value_t> src_dst_from_edge_id_and_type(
    raft::handle_t const& handle,
    rmm::device_uvector<uint64_t>& keys_to_lookup,
    bool multi_gpu) const
  {
    auto value_buffer = cugraph::allocate_dataframe_buffer<value_t>(0, handle.get_stream());

    if (multi_gpu) {
      auto& comm           = handle.get_comms();
      auto const comm_size = comm.get_size();
      auto& major_comm     = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      value_buffer =
        cugraph::collect_values_for_keys(handle,
                                         key_to_src_dst_kv_sotre->view(),
                                         keys_to_lookup.begin(),
                                         keys_to_lookup.end(),
                                         cugraph::detail::compute_gpu_id_from_ext_edge_t<edge_id_t>{
                                           comm_size, major_comm_size, minor_comm_size});
    } else {
      cugraph::resize_dataframe_buffer(value_buffer, keys_to_lookup.size(), handle.get_stream());

      key_to_src_dst_kv_sotre->view().find(keys_to_lookup.begin(),
                                           keys_to_lookup.end(),
                                           cugraph::get_dataframe_buffer_begin(value_buffer),
                                           handle.get_stream());
    }

    return std::make_tuple(std::move(std::get<0>(value_buffer)),
                           std::move(std::get<1>(value_buffer)));
  }

 private:
  using container_t = cugraph::kv_store_t<uint64_t, value_t, false /*use_binary_search*/>;
  container_t* key_to_src_dst_kv_sotre{nullptr};
};

template <typename edge_id_t, typename edge_type_t, typename vertex_t, typename value_t>
lookup_container_t<edge_id_t, edge_type_t, vertex_t, value_t>::~lookup_container_t()
{
  pimpl.reset();
}

template <typename edge_id_t, typename edge_type_t, typename vertex_t, typename value_t>
lookup_container_t<edge_id_t, edge_type_t, vertex_t, value_t>::lookup_container_t()
  : pimpl{std::make_unique<lookup_container_impl<edge_id_t, edge_type_t, vertex_t, value_t>>()}
{
}

template <typename edge_id_t, typename edge_type_t, typename vertex_t, typename value_t>
lookup_container_t<edge_id_t, edge_type_t, vertex_t, value_t>::lookup_container_t(
  raft::handle_t const& handle, size_t capacity)
  : pimpl{std::make_unique<lookup_container_impl<edge_id_t, edge_type_t, vertex_t, value_t>>(
      handle, capacity)}
{
}

template <typename edge_id_t, typename edge_type_t, typename vertex_t, typename value_t>
lookup_container_t<edge_id_t, edge_type_t, vertex_t, value_t>::lookup_container_t(
  const lookup_container_t&)
{
}

template <typename edge_id_t, typename edge_type_t, typename vertex_t, typename value_t>
void lookup_container_t<edge_id_t, edge_type_t, vertex_t, value_t>::insert(
  raft::handle_t const& handle,
  raft::device_span<edge_id_t const> edge_ids_to_insert,
  raft::device_span<edge_type_t const> edge_types_to_insert,
  dataframe_buffer_type_t<value_t>&& values_to_insert)
{
  int type_width = 12;
  rmm::device_uvector<uint64_t> keys_to_insert =
    rmm::device_uvector<uint64_t>(edge_ids_to_insert.size(), handle.get_stream());

  thrust::transform(handle.get_thrust_policy(),
                    thrust::make_zip_iterator(
                      thrust::make_tuple(edge_ids_to_insert.begin(), edge_types_to_insert.begin())),
                    thrust::make_zip_iterator(
                      thrust::make_tuple(edge_ids_to_insert.end(), edge_types_to_insert.end())),
                    keys_to_insert.begin(),
                    [composite_key_func =
                       cugraph::detail::compute_key_from_edge_id_and_type_t<edge_id_t, edge_type_t>{
                         type_width}] __device__(auto id_and_type) {
                      auto id   = thrust::get<0>(id_and_type);
                      auto type = thrust::get<1>(id_and_type);
                      return composite_key_func(id, type);
                    });

  pimpl->insert(handle, keys_to_insert, std::move(values_to_insert));
}

template <typename edge_id_t, typename edge_type_t, typename vertex_t, typename value_t>
dataframe_buffer_type_t<value_t>
lookup_container_t<edge_id_t, edge_type_t, vertex_t, value_t>::src_dst_from_edge_id_and_type(
  raft::handle_t const& handle,
  raft::device_span<edge_id_t const> edge_ids_to_lookup,
  raft::device_span<edge_type_t const> edge_types_to_lookup,
  bool multi_gpu) const
{
  int type_width = 12;
  rmm::device_uvector<uint64_t> keys_to_lookup =
    rmm::device_uvector<uint64_t>(edge_ids_to_lookup.size(), handle.get_stream());

  thrust::transform(handle.get_thrust_policy(),
                    thrust::make_zip_iterator(
                      thrust::make_tuple(edge_ids_to_lookup.begin(), edge_types_to_lookup.begin())),
                    thrust::make_zip_iterator(
                      thrust::make_tuple(edge_ids_to_lookup.end(), edge_types_to_lookup.end())),
                    keys_to_lookup.begin(),
                    [composite_key_func =
                       cugraph::detail::compute_key_from_edge_id_and_type_t<edge_id_t, edge_type_t>{
                         type_width}] __device__(auto id_and_type) {
                      auto id   = thrust::get<0>(id_and_type);
                      auto type = thrust::get<1>(id_and_type);
                      return composite_key_func(id, type);
                    });

  return pimpl->src_dst_from_edge_id_and_type(handle, keys_to_lookup, multi_gpu);
}

namespace detail {

template <typename GraphViewType,
          typename EdgeIdInputWrapper,
          typename EdgeTypeInputWrapper,
          typename EdgeTypeAndIdToSrcDstLookupContainerType>
EdgeTypeAndIdToSrcDstLookupContainerType build_edge_id_and_type_to_src_dst_lookup_map(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  EdgeIdInputWrapper edge_id_view,
  EdgeTypeInputWrapper edge_type_view)
{
  static_assert(!std::is_same_v<typename EdgeIdInputWrapper::value_type, thrust::nullopt_t>,
                "Can not create edge id lookup table without edge ids");

  using vertex_t    = typename GraphViewType::vertex_type;
  using edge_t      = typename GraphViewType::edge_type;
  using edge_type_t = typename EdgeTypeInputWrapper::value_type;
  using edge_id_t   = typename EdgeIdInputWrapper::value_type;
  using value_t     = typename EdgeTypeAndIdToSrcDstLookupContainerType::value_type;

  constexpr bool multi_gpu = GraphViewType::is_multi_gpu;
  static_assert(std::is_integral_v<edge_type_t>);
  static_assert(std::is_integral_v<edge_id_t>);
  static_assert(std::is_same_v<edge_t, edge_id_t>);
  static_assert(std::is_same_v<value_t, thrust::tuple<vertex_t, vertex_t>>);

  static_assert(
    std::is_same_v<typename EdgeTypeAndIdToSrcDstLookupContainerType::edge_type_type, edge_type_t>,
    "edge_type_t must match with EdgeTypeAndIdToSrcDstLookupContainerType::edge_type_type");

  static_assert(
    std::is_same_v<typename EdgeTypeAndIdToSrcDstLookupContainerType::edge_id_type, edge_id_t>,
    "edge_id_t must match with typename EdgeTypeAndIdToSrcDstLookupContainerType::edge_id_type");

  size_t capacity{0};
  int type_width = 12;

  if constexpr (multi_gpu) {
    auto& comm                 = handle.get_comms();
    auto const comm_size       = comm.get_size();
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();

    auto gpu_ids = cugraph::extract_transform_e(
      handle,
      graph_view,
      cugraph::edge_src_dummy_property_t{}.view(),
      cugraph::edge_dst_dummy_property_t{}.view(),
      view_concat(edge_id_view, edge_type_view),
      cuda::proclaim_return_type<thrust::optional<int>>(
        [key_mapping_func =
           cugraph::detail::compute_gpu_id_from_ext_edge_t<edge_t>{
             comm_size, major_comm_size, minor_comm_size},
         composite_key_func =
           cugraph::detail::compute_key_from_edge_id_and_type_t<edge_t, edge_type_t>{
             type_width}] __device__(auto,
                                     auto,
                                     thrust::nullopt_t,
                                     thrust::nullopt_t,
                                     thrust::tuple<edge_t, edge_type_t> id_and_type) {
          edge_t id        = thrust::get<0>(id_and_type);
          edge_type_t type = thrust::get<1>(id_and_type);

          return thrust::optional<int>{key_mapping_func(composite_key_func(id, type))};
        }));

    thrust::sort(handle.get_thrust_policy(), gpu_ids.begin(), gpu_ids.end());

    auto nr_unique_gpu_ids =
      thrust::count_if(handle.get_thrust_policy(),
                       thrust::make_counting_iterator(size_t{0}),
                       thrust::make_counting_iterator(gpu_ids.size()),
                       detail::is_first_in_run_t<decltype(gpu_ids.begin())>{gpu_ids.begin()});

    rmm::device_uvector<int> unique_gpu_ids(nr_unique_gpu_ids, handle.get_stream());

    rmm::device_uvector<edge_t> unique_gpu_id_counts(nr_unique_gpu_ids, handle.get_stream());

    thrust::reduce_by_key(handle.get_thrust_policy(),
                          gpu_ids.begin(),
                          gpu_ids.end(),
                          thrust::make_constant_iterator(size_t{1}),
                          unique_gpu_ids.begin(),
                          unique_gpu_id_counts.begin());

    gpu_ids.resize(0, handle.get_stream());
    gpu_ids.shrink_to_fit(handle.get_stream());

    std::forward_as_tuple(std::tie(unique_gpu_ids, unique_gpu_id_counts), std::ignore) =
      cugraph::groupby_gpu_id_and_shuffle_values(
        handle.get_comms(),
        thrust::make_zip_iterator(
          thrust::make_tuple(unique_gpu_ids.begin(), unique_gpu_id_counts.begin())),
        thrust::make_zip_iterator(
          thrust::make_tuple(unique_gpu_ids.end(), unique_gpu_id_counts.end())),
        [] __device__(auto val) { return static_cast<int>(thrust::get<0>(val)); },
        handle.get_stream());

    //
    // Count local #elments for all the types mapped to this GPU
    //

    capacity = static_cast<size_t>(thrust::reduce(
      handle.get_thrust_policy(), unique_gpu_id_counts.begin(), unique_gpu_id_counts.end()));

  } else {
    capacity = static_cast<size_t>(graph_view.compute_number_of_edges(handle));
  }

  auto search_container = EdgeTypeAndIdToSrcDstLookupContainerType(handle, capacity);

  //
  // Populate the search container
  //

  for (size_t local_ep_idx = 0; local_ep_idx < graph_view.number_of_local_edge_partitions();
       ++local_ep_idx) {
    //
    // decompress one edge_partition at a time
    //

    auto number_of_local_edges =
      graph_view.local_edge_partition_view(local_ep_idx).number_of_edges();

    if (graph_view.has_edge_mask()) {
      number_of_local_edges =
        detail::count_set_bits(handle,
                               (*(graph_view.edge_mask_view())).value_firsts()[local_ep_idx],
                               number_of_local_edges);
    }

    rmm::device_uvector<vertex_t> edgelist_majors(number_of_local_edges, handle.get_stream());
    rmm::device_uvector<vertex_t> edgelist_minors(edgelist_majors.size(), handle.get_stream());
    auto edgelist_ids = rmm::device_uvector<edge_t>(edgelist_majors.size(), handle.get_stream());
    auto edgelist_types =
      rmm::device_uvector<edge_type_t>(edgelist_majors.size(), handle.get_stream());

    detail::decompress_edge_partition_to_edgelist<vertex_t, edge_t, float, edge_type_t, multi_gpu>(
      handle,
      edge_partition_device_view_t<vertex_t, edge_t, multi_gpu>(
        graph_view.local_edge_partition_view(local_ep_idx)),
      std::nullopt,
      std::make_optional<detail::edge_partition_edge_property_device_view_t<edge_t, edge_t const*>>(
        edge_id_view, local_ep_idx),
      std::make_optional<
        detail::edge_partition_edge_property_device_view_t<edge_t, edge_type_t const*>>(
        edge_type_view, local_ep_idx),
      graph_view.has_edge_mask()
        ? std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *(graph_view.edge_mask_view()), local_ep_idx)
        : std::nullopt,
      raft::device_span<vertex_t>(edgelist_majors.data(), number_of_local_edges),
      raft::device_span<vertex_t>(edgelist_minors.data(), number_of_local_edges),
      std::nullopt,
      std::make_optional<raft::device_span<edge_t>>(edgelist_ids.data(), number_of_local_edges),
      std::make_optional<raft::device_span<edge_type_t>>(edgelist_types.data(),
                                                         number_of_local_edges),
      graph_view.local_edge_partition_segment_offsets(local_ep_idx));

    //
    // Shuffle to the right GPUs using function(edge id, edge_type) as keys
    //

    if constexpr (multi_gpu) {
      auto const comm_size = handle.get_comms().get_size();
      auto& major_comm     = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      // Shuffle to the proper GPUs
      std::forward_as_tuple(
        std::tie(edgelist_majors, edgelist_minors, edgelist_ids, edgelist_types), std::ignore) =
        cugraph::groupby_gpu_id_and_shuffle_values(
          handle.get_comms(),
          thrust::make_zip_iterator(thrust::make_tuple(edgelist_majors.begin(),
                                                       edgelist_minors.begin(),
                                                       edgelist_ids.begin(),
                                                       edgelist_types.begin())),
          thrust::make_zip_iterator(thrust::make_tuple(edgelist_majors.end(),
                                                       edgelist_minors.end(),
                                                       edgelist_ids.end(),
                                                       edgelist_types.end())),
          [key_mapping_func =
             cugraph::detail::compute_gpu_id_from_ext_edge_t<edge_t>{
               comm_size, major_comm_size, minor_comm_size},
           composite_key_func =
             cugraph::detail::compute_key_from_edge_id_and_type_t<edge_t, edge_type_t>{
               type_width}] __device__(auto val) {
            return key_mapping_func(composite_key_func(thrust::get<2>(val), thrust::get<3>(val)));
          },
          handle.get_stream());
    }

    auto nr_elements_to_insert = edgelist_minors.size();
    auto values_to_insert =
      cugraph::allocate_dataframe_buffer<value_t>(nr_elements_to_insert, handle.get_stream());

    auto zip_itr = thrust::make_zip_iterator(
      thrust::make_tuple(edgelist_majors.begin(), edgelist_minors.begin()));

    thrust::copy(handle.get_thrust_policy(),
                 zip_itr,
                 zip_itr + nr_elements_to_insert,
                 cugraph::get_dataframe_buffer_begin(values_to_insert));

    static_assert(
      std::is_same_v<typename thrust::iterator_traits<decltype(cugraph::get_dataframe_buffer_begin(
                       values_to_insert))>::value_type,
                     value_t>);

    search_container.insert(
      handle,
      raft::device_span<edge_t>(edgelist_ids.begin(), edgelist_ids.size()),
      raft::device_span<edge_type_t>(edgelist_types.begin(), edgelist_types.size()),
      std::move(values_to_insert));
  }

  return search_container;
}

/*
template <typename vertex_t,
          typename edge_id_t,
          typename edge_type_t,
          typename EdgeTypeAndIdToSrcDstLookupContainerType,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
lookup_endpoints_from_edge_ids_and_single_type(
  raft::handle_t const& handle,
  EdgeTypeAndIdToSrcDstLookupContainerType const& search_container,
  raft::device_span<edge_id_t const> edge_ids_to_lookup,
  edge_type_t edge_type_to_lookup)
{
  using value_t = typename EdgeTypeAndIdToSrcDstLookupContainerType::value_type;
  static_assert(std::is_integral_v<edge_id_t>);
  static_assert(std::is_integral_v<edge_type_t>);
  static_assert(std::is_same_v<value_t, thrust::tuple<vertex_t, vertex_t>>);

  static_assert(
    std::is_same_v<typename EdgeTypeAndIdToSrcDstLookupContainerType::edge_id_type, edge_id_t>,
    "edge_id_t must match EdgeTypeAndIdToSrcDstLookupContainerType::edge_id_type");
  static_assert(
    std::is_same_v<typename EdgeTypeAndIdToSrcDstLookupContainerType::edge_type_type, edge_type_t>,
    "edge_type_t must match EdgeTypeAndIdToSrcDstLookupContainerType::edge_type_type ");

  auto value_buffer = search_container.src_dst_from_edge_id_and_type(
    handle, edge_ids_to_lookup, edge_type_to_lookup, multi_gpu);

  return std::make_tuple(std::move(std::get<0>(value_buffer)),
                         std::move(std::get<1>(value_buffer)));
}
*/

template <typename vertex_t,
          typename edge_id_t,
          typename edge_type_t,
          typename EdgeTypeAndIdToSrcDstLookupContainerType,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
lookup_endpoints_from_edge_ids_and_types(
  raft::handle_t const& handle,
  EdgeTypeAndIdToSrcDstLookupContainerType const& search_container,
  raft::device_span<edge_id_t const> edge_ids_to_lookup,
  raft::device_span<edge_type_t const> edge_types_to_lookup)
{
  using value_t = typename EdgeTypeAndIdToSrcDstLookupContainerType::value_type;
  static_assert(std::is_integral_v<edge_id_t>);
  static_assert(std::is_integral_v<edge_type_t>);
  static_assert(std::is_same_v<value_t, thrust::tuple<vertex_t, vertex_t>>);

  assert(edge_ids_to_lookup.size() == edge_types_to_lookup.size());

  static_assert(
    std::is_same_v<typename EdgeTypeAndIdToSrcDstLookupContainerType::edge_id_type, edge_id_t>,
    "edge_id_t must match EdgeTypeAndIdToSrcDstLookupContainerType::edge_id_type");
  static_assert(
    std::is_same_v<typename EdgeTypeAndIdToSrcDstLookupContainerType::edge_type_type, edge_type_t>,
    "edge_type_t must match EdgeTypeAndIdToSrcDstLookupContainerType::edge_type_type ");

  auto value_buffer = search_container.src_dst_from_edge_id_and_type(
    handle, edge_ids_to_lookup, edge_types_to_lookup, multi_gpu);

  return std::make_tuple(std::move(std::get<0>(value_buffer)),
                         std::move(std::get<1>(value_buffer)));
}
}  // namespace detail

/*
template <typename vertex_t, typename edge_id_t, typename edge_type_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
lookup_endpoints_from_edge_ids_and_single_type(
  raft::handle_t const& handle,
  lookup_container_t<edge_id_t, edge_type_t, vertex_t> const& search_container,
  raft::device_span<edge_id_t const> edge_ids_to_lookup,
  edge_type_t edge_type_to_lookup)
{
  using m_t = lookup_container_t<edge_id_t, edge_type_t, vertex_t>;
  return detail::lookup_endpoints_from_edge_ids_and_single_type<vertex_t,
                                                                edge_id_t,
                                                                edge_type_t,
                                                                m_t,
                                                                multi_gpu>(
    handle, search_container, edge_ids_to_lookup, edge_type_to_lookup);
}
*/

template <typename vertex_t, typename edge_id_t, typename edge_type_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
lookup_endpoints_from_edge_ids_and_types(
  raft::handle_t const& handle,
  lookup_container_t<edge_id_t, edge_type_t, vertex_t> const& search_container,
  raft::device_span<edge_id_t const> edge_ids_to_lookup,
  raft::device_span<edge_type_t const> edge_types_to_lookup)
{
  using m_t = lookup_container_t<edge_id_t, edge_type_t, vertex_t>;
  return detail::
    lookup_endpoints_from_edge_ids_and_types<vertex_t, edge_id_t, edge_type_t, m_t, multi_gpu>(
      handle, search_container, edge_ids_to_lookup, edge_types_to_lookup);
}

template <typename vertex_t, typename edge_t, typename edge_type_t, bool multi_gpu>
lookup_container_t<edge_t, edge_type_t, vertex_t> build_edge_id_and_type_to_src_dst_lookup_map(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, edge_t const*> edge_id_view,
  edge_property_view_t<edge_t, edge_type_t const*> edge_type_view)
{
  using graph_view_t = graph_view_t<vertex_t, edge_t, false, multi_gpu>;
  using return_t     = lookup_container_t<edge_t, edge_type_t, vertex_t>;

  return detail::build_edge_id_and_type_to_src_dst_lookup_map<graph_view_t,
                                                              decltype(edge_id_view),
                                                              decltype(edge_type_view),
                                                              return_t>(
    handle, graph_view, edge_id_view, edge_type_view);
}
}  // namespace cugraph
