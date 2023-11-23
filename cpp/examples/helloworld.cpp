#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>

#include <cugraph/partition_manager.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "iostream"
using namespace std;

std::unique_ptr<raft::handle_t> initialize_mg_handle(size_t pool_size = 64)
{
  std::unique_ptr<raft::handle_t> handle{nullptr};

  handle = std::make_unique<raft::handle_t>(rmm::cuda_stream_per_thread,
                                            std::make_shared<rmm::cuda_stream_pool>(pool_size));

  raft::comms::initialize_mpi_comms(handle.get(), MPI_COMM_WORLD);
  auto& comm           = handle->get_comms();
  auto const comm_size = comm.get_size();

  std::cout << comm.get_rank() << " " << comm_size << std::endl;

  auto gpu_row_comm_size = static_cast<int>(sqrt(static_cast<double>(comm_size)));
  while (comm_size % gpu_row_comm_size != 0) {
    --gpu_row_comm_size;
  }

  cugraph::partition_manager::init_subcomm(*handle, gpu_row_comm_size);

  return std::move(handle);
}

void run_graph_algos(raft::handle_t const& handle)
{
  constexpr bool multi_gpu = true;

  constexpr bool store_transposed = false;
  constexpr bool renumber         = true;

  constexpr bool is_symmetric = true;

  constexpr bool is_weighted = true;

  size_t num_edges    = 16;
  size_t num_vertices = 6;
  size_t max_level    = 10;

  using vertex_t = int32_t;
  using edge_t   = int32_t;
  using weight_t = float;

  weight_t threshold  = 1e-7;
  weight_t resolution = 1.0;

  std::vector<vertex_t> input_src_v = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  std::vector<vertex_t> input_dst_v = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  std::vector<weight_t> input_wgt_v = {
    0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};

  std::cout << input_src_v.size() << " " << input_dst_v.size() << " " << input_wgt_v.size()
            << std::endl;

  auto const comm_rank = handle.get_comms().get_rank();
  auto const comm_size = handle.get_comms().get_size();

  auto start = comm_rank * (num_edges / comm_size) +
               (comm_rank < (num_edges % comm_size) ? comm_rank : num_edges % comm_size);

  auto end = start + (num_edges / comm_size) + ((comm_rank + 1) <= (num_edges % comm_size) ? 1 : 0);

  auto work_size = end - start;

  std::cout << start << " ---" << end << std::endl;

  std::cout << "\ncomm_rank: " << comm_rank << " comm_size: " << comm_size << std::endl;

  rmm::device_uvector<vertex_t> d_src_v(work_size, handle.get_stream());
  rmm::device_uvector<vertex_t> d_dst_v(work_size, handle.get_stream());

  auto d_wgt_v =
    is_weighted ? std::make_optional<rmm::device_uvector<weight_t>>(work_size, handle.get_stream())
                : std::nullopt;
  if (d_wgt_v) {
    raft::update_device(
      (*d_wgt_v).data(), input_wgt_v.data() + start, work_size, handle.get_stream());
  }

  raft::update_device(d_src_v.data(), input_src_v.data() + start, work_size, handle.get_stream());
  raft::update_device(d_dst_v.data(), input_dst_v.data() + start, work_size, handle.get_stream());

  for (size_t i = 0; i < comm_size; i++) {
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if (comm_rank == i) {
      std::cout << "rank " << i << " : " << std::endl;
      raft::print_device_vector("d_src_v:", d_src_v.data(), d_src_v.size(), std::cout);
      raft::print_device_vector("d_dst_v:", d_dst_v.data(), d_dst_v.size(), std::cout);
      if (d_wgt_v)
        raft::print_device_vector("d_wgt_v:", d_wgt_v->data(), d_wgt_v->size(), std::cout);
    }
  }

  //

  if (multi_gpu) {
    std::tie(store_transposed ? d_dst_v : d_src_v,
             store_transposed ? d_src_v : d_dst_v,
             d_wgt_v,
             std::ignore,
             std::ignore) =
      cugraph::detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<
        vertex_t,
        vertex_t,
        weight_t,
        int32_t>(handle,
                 store_transposed ? std::move(d_dst_v) : std::move(d_src_v),
                 store_transposed ? std::move(d_src_v) : std::move(d_dst_v),
                 std::move(d_wgt_v),
                 std::nullopt,
                 std::nullopt);
  }

  cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu> graph(handle);

  std::optional<cugraph::edge_property_t<decltype(graph.view()), weight_t>> edge_weights{
    std::nullopt};

  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};

  std::tie(graph, edge_weights, std::ignore, std::ignore, renumber_map) =
    cugraph::create_graph_from_edgelist<vertex_t,
                                        edge_t,
                                        weight_t,
                                        edge_t,
                                        int32_t,
                                        store_transposed,
                                        multi_gpu>(handle,
                                                   std::nullopt,
                                                   std::move(d_src_v),
                                                   std::move(d_dst_v),
                                                   std::move(d_wgt_v),
                                                   std::nullopt,
                                                   std::nullopt,
                                                   cugraph::graph_properties_t{is_symmetric, false},
                                                   renumber,
                                                   true);

  ///
  auto graph_view       = graph.view();
  auto edge_weight_view = edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

  rmm::device_uvector<vertex_t> d_sg_cluster_v(graph.number_of_vertices(), handle.get_stream());

  weight_t modularity{-1.0};
  std::tie(std::ignore, modularity) = cugraph::louvain(
    handle, graph_view, edge_weight_view, d_sg_cluster_v.data(), max_level, threshold, resolution);

  for (size_t i = 0; i < comm_size; i++) {
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if (comm_rank == i) { std::cout << comm_rank << " : " << modularity << std::endl; }
  }
}

int main(int argc, char** argv)
{
  RAFT_MPI_TRY(MPI_Init(&argc, &argv));

  int comm_rank{};
  RAFT_MPI_TRY(MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank));

  int comm_size{};
  RAFT_MPI_TRY(MPI_Comm_size(MPI_COMM_WORLD, &comm_size));

  int num_gpus_per_node{};
  RAFT_CUDA_TRY(cudaGetDeviceCount(&num_gpus_per_node));
  RAFT_CUDA_TRY(cudaSetDevice(comm_rank % num_gpus_per_node));

  // auto resource = std::make_shared<rmm::mr::cuda_memory_resource>();
  // rmm::mr::set_current_device_resource(resource.get());

  raft::handle_t handle = *(initialize_mg_handle());

  run_graph_algos(handle);

  RAFT_MPI_TRY(MPI_Finalize());
}