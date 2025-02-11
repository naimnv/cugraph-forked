/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <sampling/detail/sampling_utils_impl.cuh>

namespace cugraph {
namespace detail {

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<float>,
                    rmm::device_uvector<int32_t>>
count_and_remove_duplicates(raft::handle_t const& handle,
                            rmm::device_uvector<int32_t>&& src,
                            rmm::device_uvector<int32_t>&& dst,
                            rmm::device_uvector<float>&& wgt);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<float>,
                    rmm::device_uvector<int64_t>>
count_and_remove_duplicates(raft::handle_t const& handle,
                            rmm::device_uvector<int32_t>&& src,
                            rmm::device_uvector<int32_t>&& dst,
                            rmm::device_uvector<float>&& wgt);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    rmm::device_uvector<float>,
                    rmm::device_uvector<int64_t>>
count_and_remove_duplicates(raft::handle_t const& handle,
                            rmm::device_uvector<int64_t>&& src,
                            rmm::device_uvector<int64_t>&& dst,
                            rmm::device_uvector<float>&& wgt);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<double>,
                    rmm::device_uvector<int32_t>>
count_and_remove_duplicates(raft::handle_t const& handle,
                            rmm::device_uvector<int32_t>&& src,
                            rmm::device_uvector<int32_t>&& dst,
                            rmm::device_uvector<double>&& wgt);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<double>,
                    rmm::device_uvector<int64_t>>
count_and_remove_duplicates(raft::handle_t const& handle,
                            rmm::device_uvector<int32_t>&& src,
                            rmm::device_uvector<int32_t>&& dst,
                            rmm::device_uvector<double>&& wgt);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    rmm::device_uvector<double>,
                    rmm::device_uvector<int64_t>>
count_and_remove_duplicates(raft::handle_t const& handle,
                            rmm::device_uvector<int64_t>&& src,
                            rmm::device_uvector<int64_t>&& dst,
                            rmm::device_uvector<double>&& wgt);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>>
gather_one_hop_edgelist(raft::handle_t const& handle,
                        graph_view_t<int32_t, int32_t, float, false, false> const& graph_view,
                        rmm::device_uvector<int32_t> const& active_majors,
                        bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>>
gather_one_hop_edgelist(raft::handle_t const& handle,
                        graph_view_t<int32_t, int64_t, float, false, false> const& graph_view,
                        rmm::device_uvector<int32_t> const& active_majors,
                        bool do_expensive_check);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>>
gather_one_hop_edgelist(raft::handle_t const& handle,
                        graph_view_t<int64_t, int64_t, float, false, false> const& graph_view,
                        rmm::device_uvector<int64_t> const& active_majors,
                        bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>>
gather_one_hop_edgelist(raft::handle_t const& handle,
                        graph_view_t<int32_t, int32_t, double, false, false> const& graph_view,
                        rmm::device_uvector<int32_t> const& active_majors,
                        bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>>
gather_one_hop_edgelist(raft::handle_t const& handle,
                        graph_view_t<int32_t, int64_t, double, false, false> const& graph_view,
                        rmm::device_uvector<int32_t> const& active_majors,
                        bool do_expensive_check);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>>
gather_one_hop_edgelist(raft::handle_t const& handle,
                        graph_view_t<int64_t, int64_t, double, false, false> const& graph_view,
                        rmm::device_uvector<int64_t> const& active_majors,
                        bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>>
sample_edges(raft::handle_t const& handle,
             graph_view_t<int32_t, int32_t, float, false, false> const& graph_view,
             raft::random::RngState& rng_state,
             rmm::device_uvector<int32_t> const& active_majors,
             size_t fanout,
             bool with_replacement);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>>
sample_edges(raft::handle_t const& handle,
             graph_view_t<int32_t, int64_t, float, false, false> const& graph_view,
             raft::random::RngState& rng_state,
             rmm::device_uvector<int32_t> const& active_majors,
             size_t fanout,
             bool with_replacement);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>>
sample_edges(raft::handle_t const& handle,
             graph_view_t<int64_t, int64_t, float, false, false> const& graph_view,
             raft::random::RngState& rng_state,
             rmm::device_uvector<int64_t> const& active_majors,
             size_t fanout,
             bool with_replacement);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>>
sample_edges(raft::handle_t const& handle,
             graph_view_t<int32_t, int32_t, double, false, false> const& graph_view,
             raft::random::RngState& rng_state,
             rmm::device_uvector<int32_t> const& active_majors,
             size_t fanout,
             bool with_replacement);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>>
sample_edges(raft::handle_t const& handle,
             graph_view_t<int32_t, int64_t, double, false, false> const& graph_view,
             raft::random::RngState& rng_state,
             rmm::device_uvector<int32_t> const& active_majors,
             size_t fanout,
             bool with_replacement);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>>
sample_edges(raft::handle_t const& handle,
             graph_view_t<int64_t, int64_t, double, false, false> const& graph_view,
             raft::random::RngState& rng_state,
             rmm::device_uvector<int64_t> const& active_majors,
             size_t fanout,
             bool with_replacement);

}  // namespace detail
}  // namespace cugraph
