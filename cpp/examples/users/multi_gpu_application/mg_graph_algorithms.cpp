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

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <iostream>
#include <string>

void initialize_mpi_and_set_device(int argc, char** argv)
{
  RAFT_MPI_TRY(MPI_Init(&argc, &argv));

  int comm_rank{};
  RAFT_MPI_TRY(MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank));

  int num_gpus_per_node{};
  RAFT_CUDA_TRY(cudaGetDeviceCount(&num_gpus_per_node));
  RAFT_CUDA_TRY(cudaSetDevice(comm_rank % num_gpus_per_node));
}

std::unique_ptr<raft::handle_t> initialize_mg_handle(std::string const& allocation_mode = "cuda")
{
  int comm_rank{};
  RAFT_MPI_TRY(MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank));

  std::set<std::string> possible_allocation_modes = {"cuda", "pool", "binning", "managed"};

  if (possible_allocation_modes.find(allocation_mode) == possible_allocation_modes.end()) {
    if (comm_rank == 0) {
      std::cout << "'" << allocation_mode
                << "' is not a valid allocation mode. It must be one of the followings -"
                << std::endl;
      std::for_each(possible_allocation_modes.cbegin(),
                    possible_allocation_modes.cend(),
                    [](std::string mode) { std::cout << mode << std::endl; });
    }
    RAFT_MPI_TRY(MPI_Finalize());

    exit(0);
  }

  if (comm_rank == 0) {
    std::cout << "Using '" << allocation_mode
              << "' allocation mode to create device memory resources." << std::endl;
  }
  std::shared_ptr<rmm::mr::device_memory_resource> resource =
    cugraph::test::create_memory_resource(allocation_mode);
  rmm::mr::set_current_device_resource(resource.get());

  std::unique_ptr<raft::handle_t> handle =
    std::make_unique<raft::handle_t>(rmm::cuda_stream_per_thread, resource);

  raft::comms::initialize_mpi_comms(handle.get(), MPI_COMM_WORLD);
  auto& comm           = handle->get_comms();
  auto const comm_size = comm.get_size();

  auto gpu_row_comm_size = static_cast<int>(sqrt(static_cast<double>(comm_size)));
  while (comm_size % gpu_row_comm_size != 0) {
    --gpu_row_comm_size;
  }

  cugraph::partition_manager::init_subcomm(*handle, gpu_row_comm_size);

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
  auto const comm_rank = handle.get_comms().get_rank();
  auto const comm_size = handle.get_comms().get_size();

  std::cout << "Rank_" << comm_rank << ", reading graph from " << csv_graph_file_path << std::endl;
  bool renumber                            = true;  // must be true for distributed graph.
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

  auto distances_title = std::string("distances").append(std::to_string(comm_rank));
  raft::print_device_vector(distances_title.c_str(),
                            d_distances.begin(),
                            std::min<size_t>(d_distances.size(), max_nr_of_elements_to_print),
                            std::cout);

  auto predecessors_title = std::string("predecessors").append(std::to_string(comm_rank));
  raft::print_device_vector(predecessors_title.c_str(),
                            d_predecessors.begin(),
                            std::min<size_t>(d_predecessors.size(), max_nr_of_elements_to_print),
                            std::cout);

  //
  // Louvain
  //

  rmm::device_uvector<vertex_t> d_cluster_assignments(
    graph_view.local_vertex_partition_range_size(), handle.get_stream());

  weight_t threshold  = 1e-7;
  weight_t resolution = 1.0;
  size_t max_level    = 10;

  weight_t modularity{-1.0};
  std::tie(std::ignore, modularity) =
    cugraph::louvain(handle,
                     std::optional<std::reference_wrapper<raft::random::RngState>>{std::nullopt},
                     graph_view,
                     edge_weight_view,
                     d_cluster_assignments.data(),
                     max_level,
                     threshold,
                     resolution);

  //
  // Print cluster assignments of vertices
  //

  auto cluster_assignments_title =
    std::string("cluster_assignments_").append(std::to_string(comm_rank));
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  raft::print_device_vector(
    cluster_assignments_title.c_str(),
    d_cluster_assignments.begin(),
    std::min<size_t>(d_cluster_assignments.size(), max_nr_of_elements_to_print),
    std::cout);

  std::cout << "rank : " << comm_rank << ", modularity : " << modularity << std::endl;
}

int main(int argc, char** argv)
{
  if (argc < 2) {
    std::cout << "Usage: ./sg_examples path_to_your_csv_graph_file [memory allocation mode]"
              << std::endl;
    exit(0);
  }

  std::string const& csv_graph_file_path = argv[1];
  std::string const& allocation_mode     = argc < 3 ? "cuda" : argv[2];

  initialize_mpi_and_set_device(argc, argv);
  std::unique_ptr<raft::handle_t> handle = initialize_mg_handle(allocation_mode);

  auto const comm_rank = handle->get_comms().get_rank();
  auto const comm_size = handle->get_comms().get_size();

  using vertex_t                  = int32_t;
  using edge_t                    = int32_t;
  using weight_t                  = float;
  constexpr bool multi_gpu        = true;
  constexpr bool store_transposed = false;

  run_graph_algos<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
    *handle, csv_graph_file_path, true);
}
