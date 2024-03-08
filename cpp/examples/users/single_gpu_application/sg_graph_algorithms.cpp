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

#include "../tests/utilities/base_fixture.hpp"
#include "../tests/utilities/test_utilities.hpp"

#include <cugraph/algorithms.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <iostream>
#include <string>

std::unique_ptr<raft::handle_t> initialize_sg_handle(std::string const& allocation_mode = "cuda")
{
  std::set<std::string> possible_allocation_modes = {"cuda", "pool", "binning", "managed"};

  if (possible_allocation_modes.find(allocation_mode) == possible_allocation_modes.end()) {
    std::cout << "'" << allocation_mode
              << "' is not a valid allocation mode. It must be one of the followings -"
              << std::endl;
    std::for_each(possible_allocation_modes.cbegin(),
                  possible_allocation_modes.cend(),
                  [](std::string mode) { std::cout << mode << std::endl; });

    exit(0);
  }
  std::cout << "Using '" << allocation_mode
            << "' allocation mode to create device memory resources." << std::endl;

  RAFT_CUDA_TRY(cudaSetDevice(0));
  std::shared_ptr<rmm::mr::device_memory_resource> resource =
    cugraph::test::create_memory_resource(allocation_mode);
  rmm::mr::set_current_device_resource(resource.get());

  std::unique_ptr<raft::handle_t> handle =
    std::make_unique<raft::handle_t>(rmm::cuda_stream_per_thread, resource);
  return std::move(handle);
}

/**
 * @brief This function reads graph from an input csv file and run BFS and Louvain on it.
 */

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
void run_graph_algos(raft::handle_t const& handle,
                     std::string const& csv_graph_file_path,
                     bool weighted = false)
{
  std::cout << "Reading graph from " << csv_graph_file_path << std::endl;
  bool renumber                            = false;  // for single gpu, this can be true or false
  auto [graph, edge_weights, renumber_map] = cugraph::test::
    read_graph_from_csv_file<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      handle, csv_graph_file_path, weighted, renumber);

  auto graph_view       = graph.view();
  auto edge_weight_view = edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

  if (renumber_map.has_value()) {
    assert(graph_view.local_vertex_partition_range_size() == (*renumber_map).size());
  }

  //
  // Run BFS and Louvain as example algorithms
  //

  rmm::device_uvector<vertex_t> d_distances(graph_view.local_vertex_partition_range_size(),
                                            handle.get_stream());
  rmm::device_uvector<vertex_t> d_predecessors(graph_view.local_vertex_partition_range_size(),
                                               handle.get_stream());

  rmm::device_uvector<vertex_t> d_sources(1, handle.get_stream());
  std::vector<vertex_t> h_sources = {0};
  raft::update_device(d_sources.data(), h_sources.data(), h_sources.size(), handle.get_stream());

  cugraph::bfs(handle,
               graph_view,
               d_distances.data(),
               d_predecessors.data(),
               d_sources.data(),
               d_sources.size(),
               false,
               std::numeric_limits<vertex_t>::max());

  size_t max_nr_of_elements_to_print = 20;
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  raft::print_device_vector("distances",
                            d_distances.begin(),
                            std::min<size_t>(d_distances.size(), max_nr_of_elements_to_print),
                            std::cout);

  raft::print_device_vector("predecessors",
                            d_predecessors.begin(),
                            std::min<size_t>(d_predecessors.size(), max_nr_of_elements_to_print),
                            std::cout);

  //
  // Louvain
  //

  rmm::device_uvector<vertex_t> cluster_assignments(graph_view.local_vertex_partition_range_size(),
                                                    handle.get_stream());

  weight_t threshold  = 1e-7;
  weight_t resolution = 1.0;
  size_t max_level    = 10;

  weight_t modularity{-1.0};
  std::tie(std::ignore, modularity) =
    cugraph::louvain(handle,
                     std::optional<std::reference_wrapper<raft::random::RngState>>{std::nullopt},
                     graph_view,
                     edge_weight_view,
                     cluster_assignments.data(),
                     max_level,
                     threshold,
                     resolution);

  //
  // Print cluster assignments of vertices
  //

  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  raft::print_device_vector(
    "cluster_assignments",
    cluster_assignments.begin(),
    std::min<size_t>(cluster_assignments.size(), max_nr_of_elements_to_print),
    std::cout);

  std::cout << "modularity : " << modularity << std::endl;
}

int main(int argc, char** argv)
{
  if (argc < 2) {
    std::cout << "Usage: ./sg_examples path_to_your_csv_graph_file [memory allocation mode]"
              << std::endl;
    exit(0);
  }

  std::string const& allocation_mode     = argc < 3 ? "cuda" : argv[2];
  std::unique_ptr<raft::handle_t> handle = initialize_sg_handle(allocation_mode);

  constexpr bool multi_gpu        = false;
  constexpr bool store_transposed = false;

  using vertex_t = int32_t;
  using edge_t   = int32_t;
  using weight_t = float;

  run_graph_algos<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(*handle, argv[1], true);
}
