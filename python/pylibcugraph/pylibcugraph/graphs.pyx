# Copyright (c) 2022, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Have cython use python 3 syntax
# cython: language_level = 3

from libc.stdint cimport uintptr_t

from pylibcugraph._cugraph_c.resource_handle cimport (
    bool_t,
    cugraph_resource_handle_t,
    data_type_id_t,
)
from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
    cugraph_type_erased_device_array_view_create,
    cugraph_type_erased_device_array_view_free,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
    cugraph_sg_graph_create,
    cugraph_mg_graph_create,
    cugraph_graph_properties_t,
    cugraph_sg_graph_free,
    cugraph_mg_graph_free,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
    cugraph_mg_graph_create,
    cugraph_graph_properties_t,
    cugraph_mg_graph_free,
)
from pylibcugraph.resource_handle cimport (
    ResourceHandle,
)
from pylibcugraph.graph_properties cimport (
    GraphProperties,
)
from pylibcugraph.utils cimport (
    assert_success,
    assert_CAI_type,
    get_c_type_from_numpy_type,
)


cdef class SGGraph(_GPUGraph):
    """
    RAII-stye Graph class for use with single-GPU APIs that manages the
    individual create/free calls and the corresponding cugraph_graph_t pointer.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph_properties : GraphProperties
        Object defining intended properties for the graph.

    src_array : device array type
        Device array containing the vertex identifiers of the source of each
        directed edge. The order of the array corresponds to the ordering of the
        dst_array, where the ith item in src_array and the ith item in dst_array
        define the ith edge of the graph.

    dst_array : device array type
        Device array containing the vertex identifiers of the destination of
        each directed edge. The order of the array corresponds to the ordering
        of the src_array, where the ith item in src_array and the ith item in
        dst_array define the ith edge of the graph.

    weight_array : device array type
        Device array containing the weight values of each directed edge. The
        order of the array corresponds to the ordering of the src_array and
        dst_array arrays, where the ith item in weight_array is the weight value
        of the ith edge of the graph.

    store_transposed : bool
        Set to True if the graph should be transposed. This is required for some
        algorithms, such as pagerank.

    renumber : bool
        Set to True to indicate the vertices used in src_array and dst_array are
        not appropriate for use as internal array indices, and should be mapped
        to continuous integers starting from 0.

    do_expensive_check : bool
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.
    
    edge_id_array : device array type
        Device array containing the edge ids of each directed edge.  Must match
        the ordering of the src/dst arrays.  Optional (may be null).  If
        provided, edge_type_array must also be provided.
    
    edge_type_array : device array type
        Device array containing the edge types of each directed edge.  Must
        match the ordering of the src/dst/edge_id arrays.  Optional (may be
        null).  If provided, edge_id_array must be provided.
    
    Examples
    ---------
    >>> import pylibcugraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 2], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 2, 3], dtype=numpy.int32)
    >>> seeds = cupy.asarray([0, 0, 1], dtype=numpy.int32)
    >>> weights = cupy.asarray([1.0, 1.0, 1.0], dtype=numpy.float32)
    >>> resource_handle = pylibcugraph.ResourceHandle()
    >>> graph_props = pylibcugraph.GraphProperties(
    ...     is_symmetric=False, is_multigraph=False)
    >>> G = pylibcugraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weights,
    ...     store_transposed=False, renumber=False, do_expensive_check=False)

    """
    def __cinit__(self,
                  ResourceHandle resource_handle,
                  GraphProperties graph_properties,
                  src_array,
                  dst_array,
                  weight_array,
                  store_transposed=False,
                  renumber=False,
                  do_expensive_check=False,
                  edge_id_array=None,
                  edge_type_array=None):

        # FIXME: add tests for these
        if not(isinstance(store_transposed, (int, bool))):
            raise TypeError("expected int or bool for store_transposed, got "
                            f"{type(store_transposed)}")
        if not(isinstance(renumber, (int, bool))):
            raise TypeError("expected int or bool for renumber, got "
                            f"{type(renumber)}")
        if not(isinstance(do_expensive_check, (int, bool))):
            raise TypeError("expected int or bool for do_expensive_check, got "
                            f"{type(do_expensive_check)}")
        assert_CAI_type(src_array, "src_array")
        assert_CAI_type(dst_array, "dst_array")
        assert_CAI_type(weight_array, "weight_array")
        if edge_id_array is not None:
            assert_CAI_type(edge_id_array, "edge_id_array")
        if edge_type_array is not None:
            assert_CAI_type(edge_type_array, "edge_type_array")

        # FIXME: assert that src_array and dst_array have the same type

        cdef cugraph_error_t* error_ptr
        cdef cugraph_error_code_t error_code

        cdef uintptr_t cai_srcs_ptr = \
            src_array.__cuda_array_interface__["data"][0]
        cdef cugraph_type_erased_device_array_view_t* srcs_view_ptr = \
            cugraph_type_erased_device_array_view_create(
                <void*>cai_srcs_ptr,
                len(src_array),
                get_c_type_from_numpy_type(src_array.dtype))

        cdef uintptr_t cai_dsts_ptr = \
            dst_array.__cuda_array_interface__["data"][0]
        cdef cugraph_type_erased_device_array_view_t* dsts_view_ptr = \
            cugraph_type_erased_device_array_view_create(
                <void*>cai_dsts_ptr,
                len(dst_array),
                get_c_type_from_numpy_type(dst_array.dtype))

        cdef uintptr_t cai_weights_ptr = \
            weight_array.__cuda_array_interface__["data"][0]
        cdef cugraph_type_erased_device_array_view_t* weights_view_ptr = \
            cugraph_type_erased_device_array_view_create(
                <void*>cai_weights_ptr,
                len(weight_array),
                get_c_type_from_numpy_type(weight_array.dtype))
        
        cdef cugraph_type_erased_device_array_view_t* edge_id_view_ptr = NULL
        cdef uintptr_t cai_edge_id_ptr
        if edge_id_array is not None:
            cai_edge_id_ptr = (
                edge_id_array.__cuda_array_interface__["data"][0]
            )
            edge_id_view_ptr = (
                cugraph_type_erased_device_array_view_create(
                    <void*>cai_edge_id_ptr,
                    len(edge_id_array),
                    get_c_type_from_numpy_type(edge_id_array.dtype)
                )
            )

        cdef cugraph_type_erased_device_array_view_t* edge_type_view_ptr = NULL
        cdef uintptr_t cai_edge_type_ptr
        if edge_type_array is not None:
            cai_edge_type_ptr = (
                edge_type_array.__cuda_array_interface__["data"][0]
            )
            edge_type_view_ptr = (
                cugraph_type_erased_device_array_view_create(
                    <void*>cai_edge_type_ptr,
                    len(edge_type_array),
                    get_c_type_from_numpy_type(edge_type_array.dtype)
                )
            )

        error_code = cugraph_sg_graph_create(
            resource_handle.c_resource_handle_ptr,
            &(graph_properties.c_graph_properties),
            srcs_view_ptr,
            dsts_view_ptr,
            weights_view_ptr,
            edge_id_view_ptr,
            edge_type_view_ptr,
            store_transposed,
            renumber,
            do_expensive_check,
            &(self.c_graph_ptr),
            &error_ptr)

        assert_success(error_code, error_ptr,
                       "cugraph_sg_graph_create()")

        cugraph_type_erased_device_array_view_free(srcs_view_ptr)
        cugraph_type_erased_device_array_view_free(dsts_view_ptr)
        cugraph_type_erased_device_array_view_free(weights_view_ptr)
        if edge_id_view_ptr is not NULL:
            cugraph_type_erased_device_array_view_free(edge_id_view_ptr)
        if edge_type_view_ptr is not NULL:
            cugraph_type_erased_device_array_view_free(edge_type_view_ptr)

    def __dealloc__(self):
        if self.c_graph_ptr is not NULL:
            cugraph_sg_graph_free(self.c_graph_ptr)


cdef class MGGraph(_GPUGraph):
    """
    RAII-stye Graph class for use with multi-GPU APIs that manages the
    individual create/free calls and the corresponding cugraph_graph_t pointer.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph_properties : GraphProperties
        Object defining intended properties for the graph.

    src_array : device array type
        Device array containing the vertex identifiers of the source of each
        directed edge. The order of the array corresponds to the ordering of the
        dst_array, where the ith item in src_array and the ith item in dst_array
        define the ith edge of the graph.

    dst_array : device array type
        Device array containing the vertex identifiers of the destination of
        each directed edge. The order of the array corresponds to the ordering
        of the src_array, where the ith item in src_array and the ith item in
        dst_array define the ith edge of the graph.

    weight_array : device array type
        Device array containing the weight values of each directed edge. The
        order of the array corresponds to the ordering of the src_array and
        dst_array arrays, where the ith item in weight_array is the weight value
        of the ith edge of the graph.

    store_transposed : bool
        Set to True if the graph should be transposed. This is required for some
        algorithms, such as pagerank.
    
    num_edges : int
        Number of edges
    
    do_expensive_check : bool
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.

    edge_id_array : device array type
        Device array containing the edge ids of each directed edge.  Must match
        the ordering of the src/dst arrays.  Optional (may be null).  If
        provided, edge_type_array must also be provided.
    
    edge_type_array : device array type
        Device array containing the edge types of each directed edge.  Must
        match the ordering of the src/dst/edge_id arrays.  Optional (may be
        null).  If provided, edge_id_array must be provided.
    """
    def __cinit__(self,
                  ResourceHandle resource_handle,
                  GraphProperties graph_properties,
                  src_array,
                  dst_array,
                  weight_array,
                  store_transposed=False,
                  num_edges=-1,
                  do_expensive_check=False,
                  edge_id_array=None,
                  edge_type_array=None):

        # FIXME: add tests for these
        if not(isinstance(store_transposed, (int, bool))):
            raise TypeError("expected int or bool for store_transposed, got "
                            f"{type(store_transposed)}")
        if not(isinstance(num_edges, (int))):
            raise TypeError("expected int for num_edges, got "
                            f"{type(num_edges)}")
        if num_edges < 0:
            raise TypeError("num_edges must be > 0")
        if not(isinstance(do_expensive_check, (int, bool))):
            raise TypeError("expected int or bool for do_expensive_check, got "
                            f"{type(do_expensive_check)}")
        assert_CAI_type(src_array, "src_array")
        assert_CAI_type(dst_array, "dst_array")
        assert_CAI_type(weight_array, "weight_array")
        if edge_id_array is not None:
            assert_CAI_type(edge_id_array, "edge_id_array")
        if edge_type_array is not None:
            assert_CAI_type(edge_type_array, "edge_type_array")        

        # FIXME: assert that src_array and dst_array have the same type

        cdef cugraph_error_t* error_ptr
        cdef cugraph_error_code_t error_code

        cdef uintptr_t cai_srcs_ptr = \
            src_array.__cuda_array_interface__["data"][0]
        cdef cugraph_type_erased_device_array_view_t* srcs_view_ptr = \
            cugraph_type_erased_device_array_view_create(
                <void*>cai_srcs_ptr,
                len(src_array),
                get_c_type_from_numpy_type(src_array.dtype))

        cdef uintptr_t cai_dsts_ptr = \
            dst_array.__cuda_array_interface__["data"][0]
        cdef cugraph_type_erased_device_array_view_t* dsts_view_ptr = \
            cugraph_type_erased_device_array_view_create(
                <void*>cai_dsts_ptr,
                len(dst_array),
                get_c_type_from_numpy_type(dst_array.dtype))

        cdef uintptr_t cai_weights_ptr = \
            weight_array.__cuda_array_interface__["data"][0]
        cdef cugraph_type_erased_device_array_view_t* weights_view_ptr = \
            cugraph_type_erased_device_array_view_create(
                <void*>cai_weights_ptr,
                len(weight_array),
                get_c_type_from_numpy_type(weight_array.dtype))

        cdef cugraph_type_erased_device_array_view_t* edge_id_view_ptr = NULL
        cdef uintptr_t cai_edge_id_ptr
        if edge_id_array is not None:
            cai_edge_id_ptr = (
                edge_id_array.__cuda_array_interface__["data"][0]
            )
            edge_id_view_ptr = (
                cugraph_type_erased_device_array_view_create(
                    <void*>cai_edge_id_ptr,
                    len(edge_id_array),
                    get_c_type_from_numpy_type(edge_id_array.dtype)
                )
            )

        cdef cugraph_type_erased_device_array_view_t* edge_type_view_ptr = NULL
        cdef uintptr_t cai_edge_type_ptr
        if edge_type_array is not None:
            cai_edge_type_ptr = (
                edge_type_array.__cuda_array_interface__["data"][0]
            )
            edge_type_view_ptr = (
                cugraph_type_erased_device_array_view_create(
                    <void*>cai_edge_type_ptr,
                    len(edge_type_array),
                    get_c_type_from_numpy_type(edge_type_array.dtype)
                )
            )

        error_code = cugraph_mg_graph_create(
            resource_handle.c_resource_handle_ptr,
            &(graph_properties.c_graph_properties),
            srcs_view_ptr,
            dsts_view_ptr,
            weights_view_ptr,
            edge_id_view_ptr,
            edge_type_view_ptr,
            store_transposed,
            num_edges,
            do_expensive_check,
            &(self.c_graph_ptr),
            &error_ptr)

        assert_success(error_code, error_ptr,
                       "cugraph_mg_graph_create()")

        cugraph_type_erased_device_array_view_free(srcs_view_ptr)
        cugraph_type_erased_device_array_view_free(dsts_view_ptr)
        cugraph_type_erased_device_array_view_free(weights_view_ptr)
        if edge_id_view_ptr is not NULL:
            cugraph_type_erased_device_array_view_free(edge_id_view_ptr)
        if edge_type_view_ptr is not NULL:
            cugraph_type_erased_device_array_view_free(edge_type_view_ptr)

    def __dealloc__(self):
        if self.c_graph_ptr is not NULL:
            cugraph_mg_graph_free(self.c_graph_ptr)
