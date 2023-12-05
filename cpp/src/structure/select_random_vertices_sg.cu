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

#include <structure/select_random_vertices_impl.hpp>

namespace cugraph {

template rmm::device_uvector<int32_t> select_random_vertices(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<raft::device_span<int32_t const>> given_set,
  raft::random::RngState& rng_state,
  size_t select_count,
  bool with_replacement,
  bool sort_vertices,
  bool shuffle_int_to_local,
  bool do_expensive_check);

template rmm::device_uvector<int32_t> select_random_vertices(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, false, false> const& graph_view,
  std::optional<raft::device_span<int32_t const>> given_set,
  raft::random::RngState& rng_state,
  size_t select_count,
  bool with_replacement,
  bool sort_vertices,
  bool shuffle_int_to_local,
  bool do_expensive_check);

template rmm::device_uvector<int64_t> select_random_vertices(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  std::optional<raft::device_span<int64_t const>> given_set,
  raft::random::RngState& rng_state,
  size_t select_count,
  bool with_replacement,
  bool sort_vertices,
  bool shuffle_int_to_local,
  bool do_expensive_check);

template rmm::device_uvector<int32_t> select_random_vertices(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  std::optional<raft::device_span<int32_t const>> given_set,
  raft::random::RngState& rng_state,
  size_t select_count,
  bool with_replacement,
  bool sort_vertices,
  bool shuffle_int_to_local,
  bool do_expensive_check);

template rmm::device_uvector<int32_t> select_random_vertices(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, true, false> const& graph_view,
  std::optional<raft::device_span<int32_t const>> given_set,
  raft::random::RngState& rng_state,
  size_t select_count,
  bool with_replacement,
  bool sort_vertices,
  bool shuffle_int_to_local,
  bool do_expensive_check);

template rmm::device_uvector<int64_t> select_random_vertices(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  std::optional<raft::device_span<int64_t const>> given_set,
  raft::random::RngState& rng_state,
  size_t select_count,
  bool with_replacement,
  bool sort_vertices,
  bool shuffle_int_to_local,
  bool do_expensive_check);

}  // namespace cugraph
