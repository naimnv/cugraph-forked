# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

from cugraph.experimental import MGPropertyGraph

from typing import Optional, Tuple, Any
from enum import Enum

import cupy
import cudf
import dask_cudf
import cugraph

from dataclasses import dataclass
from collections import defaultdict
from itertools import chain


class EdgeLayout(Enum):
    COO = 'coo'
    CSC = 'csc'
    CSR = 'csr'


@dataclass
class CuGraphEdgeAttr:
    r"""Defines the attributes of an :obj:`GraphStore` edge."""

    # The type of the edge
    edge_type: Optional[Any]

    # The layout of the edge representation
    layout: EdgeLayout

    # Whether the edge index is sorted, by destination node. Useful for
    # avoiding sorting costs when performing neighbor sampling, and only
    # meaningful for COO (CSC and CSR are sorted by definition)
    is_sorted: bool = False

    # The number of nodes in this edge type. If set to None, will attempt to
    # infer with the simple heuristic int(self.edge_index.max()) + 1
    size: Optional[Tuple[int, int]] = None

    # NOTE we define __post_init__ to force-cast layout
    def __post_init__(self):
        self.layout = EdgeLayout(self.layout)

    @classmethod
    def cast(cls, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            elem = args[0]
            if elem is None:
                return None
            if isinstance(elem, CuGraphEdgeAttr):
                return elem
            if isinstance(elem, (tuple, list)):
                return cls(*elem)
            if isinstance(elem, dict):
                return cls(**elem)
        return cls(*args, **kwargs)


def EXPERIMENTAL__to_pyg(G, backend='torch'):
    """
        Returns the PyG wrappers for the provided PropertyGraph or
        MGPropertyGraph.

    Parameters
    ----------
    G : PropertyGraph or MGPropertyGraph
        The graph to produce PyG wrappers for.

    Returns
    -------
    Tuple (CuGraphStore, CuGraphStore)
        Wrappers for the provided property graph.
    """
    store = EXPERIMENTAL__CuGraphStore(G, backend=backend)
    return (store, store)


_field_status = Enum("FieldStatus", "UNSET")


@dataclass
class CuGraphTensorAttr:
    r"""Defines the attributes of a class:`FeatureStore` tensor; in particular,
    all the parameters necessary to uniquely identify a tensor from the feature
    store.

    Note that the order of the attributes is important; this is the order in
    which attributes must be provided for indexing calls. Feature store
    implementor classes can define a different ordering by overriding
    :meth:`TensorAttr.__init__`.
    """

    # The group name that the tensor corresponds to. Defaults to UNSET.
    group_name: Optional[str] = _field_status.UNSET

    # The name of the tensor within its group. Defaults to UNSET.
    attr_name: Optional[str] = _field_status.UNSET

    # The node indices the rows of the tensor correspond to. Defaults to UNSET.
    index: Optional[Any] = _field_status.UNSET

    # The properties in the PropertyGraph the rows of the tensor correspond to.
    # Defaults to UNSET.
    properties: Optional[Any] = _field_status.UNSET

    # The datatype of the tensor.  Defaults to UNSET.
    dtype: Optional[Any] = _field_status.UNSET

    # Convenience methods

    def is_set(self, key):
        r"""Whether an attribute is set in :obj:`TensorAttr`."""
        if key not in self.__dataclass_fields__:
            raise KeyError(key)
        attr = getattr(self, key)
        return type(attr) != _field_status or attr != _field_status.UNSET

    def is_fully_specified(self):
        r"""Whether the :obj:`TensorAttr` has no unset fields."""
        return all([self.is_set(key) for key in self.__dataclass_fields__])

    def fully_specify(self):
        r"""Sets all :obj:`UNSET` fields to :obj:`None`."""
        for key in self.__dataclass_fields__:
            if not self.is_set(key):
                setattr(self, key, None)
        return self

    def update(self, attr):
        r"""Updates an :class:`TensorAttr` with set attributes from another
        :class:`TensorAttr`."""
        for key in self.__dataclass_fields__:
            if attr.is_set(key):
                setattr(self, key, getattr(attr, key))

    @classmethod
    def cast(cls, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            elem = args[0]
            if elem is None:
                return None
            if isinstance(elem, CuGraphTensorAttr):
                return elem
            if isinstance(elem, (tuple, list)):
                return cls(*elem)
            if isinstance(elem, dict):
                return cls(**elem)
        return cls(*args, **kwargs)


class EXPERIMENTAL__CuGraphStore:
    """
    Duck-typed version of PyG's GraphStore and FeatureStore.
    """
    def __init__(self, G, reserved_keys=[], backend='torch'):
        """
            G : PropertyGraph or MGPropertyGraph
                The cuGraph property graph where the
                data is being stored.
            reserved_keys : Properties in the graph that are not used for
                training (the 'x' attribute will ignore these properties).
            backend : The backend that manages tensors (default = 'torch')
                Should usually be 'torch' ('torch', 'cupy' supported).
        """

        # TODO ensure all x properties are float32 type
        # TODO ensure y is of long type
        if None in G.edge_types:
            raise ValueError('Unspecified edge types not allowed in PyG')

        if backend == 'torch':
            from torch.utils.dlpack import from_dlpack
            from torch import int64 as vertex_dtype
            from torch import float32 as property_dtype
        elif backend == 'cupy':
            from cupy import from_dlpack
            from cupy import int64 as vertex_dtype
            from cupy import float32 as property_dtype
        else:
            raise ValueError(f'Invalid backend {backend}.')
        self.__backend = backend
        self.from_dlpack = from_dlpack
        self.vertex_dtype = vertex_dtype
        self.property_dtype = property_dtype

        self.__graph = G
        self.__subgraphs = {}

        self.__reserved_keys = [
            self.__graph.type_col_name,
            self.__graph.vertex_col_name
        ] + list(reserved_keys)

        self._tensor_attr_cls = CuGraphTensorAttr
        self._tensor_attr_dict = defaultdict(list)
        self.__infer_x_and_y_tensors()

        self.__edge_types_to_attrs = {}
        for edge_type in self.__graph.edge_types:
            edges = self.__graph.get_edge_data(types=[edge_type])
            dsts = edges[self.__graph.dst_col_name].unique()
            srcs = edges[self.__graph.src_col_name].unique()

            if self.is_mg:
                dsts = dsts.compute()
                srcs = srcs.compute()

            dst_types = self.__graph.get_vertex_data(
                vertex_ids=dsts.values_host,
                columns=[self.__graph.type_col_name]
            )[self.__graph.type_col_name].unique()

            src_types = self.__graph.get_vertex_data(
                vertex_ids=srcs.values_host,
                columns=[self.__graph.type_col_name]
            )[self.__graph.type_col_name].unique()

            if self.is_mg:
                dst_types = dst_types.compute()
                src_types = src_types.compute()

            err_string = (
                f'Edge type {edge_type} associated'
                'with multiple src/dst type pairs'
            )
            if len(dst_types) > 1 or len(src_types) > 1:
                raise TypeError(err_string)

            pyg_edge_type = (src_types[0], edge_type, dst_types[0])

            self.__edge_types_to_attrs[edge_type] = CuGraphEdgeAttr(
                edge_type=pyg_edge_type,
                layout=EdgeLayout.COO,
                is_sorted=False,
                size=len(edges)
            )

            self._edge_attr_cls = CuGraphEdgeAttr

    @property
    def _edge_types_to_attrs(self):
        return dict(self.__edge_types_to_attrs)

    @property
    def backend(self):
        return self.__backend

    @property
    def is_mg(self):
        return isinstance(self.__graph, MGPropertyGraph)

    def put_edge_index(self, edge_index, edge_attr):
        raise NotImplementedError('Adding indices not supported.')

    def get_all_edge_attrs(self):
        """
            Returns all edge types and indices in this store.
        """
        return self.__edge_types_to_attrs.values()

    def _get_edge_index(self, attr):
        """
            Returns the edge index in the requested format
            (as defined by attr).  Currently, only unsorted
            COO is supported, which is returned as a (src,dst)
            tuple as expected by the PyG API.

            Parameters
            ----------
            attr: CuGraphEdgeAttr
                The CuGraphEdgeAttr specifying the
                desired edge type, layout (i.e. CSR, COO, CSC), and
                whether the returned index should be sorted (if COO).
                Currently, only unsorted COO is supported.

            Returns
            -------
            (src, dst) : Tuple[tensor type]
                Tuple of the requested edge index in COO form.
                Currently, only COO form is supported.
        """

        if attr.layout != EdgeLayout.COO:
            raise TypeError('Only COO direct access is supported!')

        if isinstance(attr.edge_type, str):
            edge_type = attr.edge_type
        else:
            edge_type = attr.edge_type[1]

        # If there is only one edge type (homogeneous graph) then
        # bypass the edge filters for a significant speed improvement.
        if len(self.__graph.edge_types) == 1:
            if list(self.__graph.edge_types)[0] != edge_type:
                raise ValueError(
                    f'Requested edge type {edge_type}'
                    'is not present in graph.'
                )

            df = self.__graph.get_edge_data(
                edge_ids=None,
                types=None,
                columns=[
                    self.__graph.src_col_name,
                    self.__graph.dst_col_name
                ]
            )
        else:
            if isinstance(attr.edge_type, str):
                edge_type = attr.edge_type
            else:
                edge_type = attr.edge_type[1]

            # FIXME unrestricted edge type names
            df = self.__graph.get_edge_data(
                edge_ids=None,
                types=[edge_type],
                columns=[
                    self.__graph.src_col_name,
                    self.__graph.dst_col_name
                ]
            )

        if self.is_mg:
            df = df.compute()

        src = self.from_dlpack(df[self.__graph.src_col_name].to_dlpack())
        dst = self.from_dlpack(df[self.__graph.dst_col_name].to_dlpack())

        if self.__backend == 'torch':
            src = src.to(self.vertex_dtype)
            dst = dst.to(self.vertex_dtype)
        elif self.__backend == 'cupy':
            src = src.astype(self.vertex_dtype)
            dst = dst.astype(self.vertex_dtype)
        else:
            raise TypeError(f'Invalid backend type {self.__backend}')

        if self.__backend == 'torch':
            src = src.to(self.vertex_dtype)
            dst = dst.to(self.vertex_dtype)
        else:
            # self.__backend == 'cupy'
            src = src.astype(self.vertex_dtype)
            dst = dst.astype(self.vertex_dtype)

        if src.shape[0] != dst.shape[0]:
            raise IndexError('src and dst shape do not match!')

        return (src, dst)

    def get_edge_index(self, *args, **kwargs):
        r"""Synchronously gets an edge_index tensor from the materialized
        graph.

        Args:
            **attr(EdgeAttr): the edge attributes.

        Returns:
            EdgeTensorType: an edge_index tensor corresonding to the provided
            attributes, or None if there is no such tensor.

        Raises:
            KeyError: if the edge index corresponding to attr was not found.
        """

        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        edge_attr.layout = EdgeLayout(edge_attr.layout)
        # Override is_sorted for CSC and CSR:
        # TODO treat is_sorted specially in this function, where is_sorted=True
        # returns an edge index sorted by column.
        edge_attr.is_sorted = edge_attr.is_sorted or (edge_attr.layout in [
            EdgeLayout.CSC, EdgeLayout.CSR
        ])
        edge_index = self._get_edge_index(edge_attr)
        if edge_index is None:
            raise KeyError(f"An edge corresponding to '{edge_attr}' was not "
                           f"found")
        return edge_index

    def _subgraph(self, edge_types):
        """
        Returns a subgraph with edges limited to those of a given type

        Parameters
        ----------
        edge_types : list of edge types
            Directly references the graph's internal edge types.  Does
            not accept PyG edge type tuples.

        Returns
        -------
        The appropriate extracted subgraph.  Will extract the subgraph
        if it has not already been extracted.

        """
        edge_types = tuple(sorted(edge_types))

        if edge_types not in self.__subgraphs:
            query = f'(_TYPE_=="{edge_types[0]}")'
            for t in edge_types[1:]:
                query += f' | (_TYPE_=="{t}")'
            selection = self.__graph.select_edges(query)

            # FIXME enforce int type
            sg = self.__graph.extract_subgraph(
                selection=selection,
                edge_weight_property=self.__graph.edge_id_col_name,
                default_edge_weight=1.0,
                check_multi_edges=True,
                renumber_graph=True,
                add_edge_data=False
            )
            self.__subgraphs[edge_types] = sg

        return self.__subgraphs[edge_types]

    def neighbor_sample(
            self,
            index,
            num_neighbors,
            replace,
            directed,
            edge_types):

        if isinstance(num_neighbors, dict):
            # FIXME support variable num neighbors per edge type
            num_neighbors = list(num_neighbors.values())[0]

        # FIXME eventually get uniform neighbor sample to accept longs
        if self.__backend == 'torch' and not index.is_cuda:
            index = index.cuda()
        index = cupy.from_dlpack(index.__dlpack__())

        # FIXME resolve the directed/undirected issue
        G = self._subgraph([et[1] for et in edge_types])

        index = cudf.Series(index)
        if self.is_mg:
            uniform_neighbor_sample = cugraph.dask.uniform_neighbor_sample
        else:
            uniform_neighbor_sample = cugraph.uniform_neighbor_sample

        sampling_results = uniform_neighbor_sample(
                G,
                index,
                # conversion required by cugraph api
                list(num_neighbors),
                replace
            )

        concat_fn = dask_cudf.concat if self.is_mg else cudf.concat

        nodes_of_interest = concat_fn(
            [sampling_results.destinations, sampling_results.sources]
        ).unique()

        if self.is_mg:
            nodes_of_interest = nodes_of_interest.compute()

        # Get the node index (for creating the edge index),
        # the node type groupings, and the node properties.
        noi_index, noi_groups, noi_tensors = (
            self.__get_renumbered_vertex_data_from_sample(
                nodes_of_interest
            )
        )

        # Get the new edge index (by type as expected for HeteroData)
        # FIXME handle edge ids
        row_dict, col_dict = self.__get_renumbered_edges_from_sample(
            sampling_results,
            noi_index
        )

        return (noi_groups, row_dict, col_dict, noi_tensors)

    def __get_renumbered_vertex_data_from_sample(self, nodes_of_interest):
        nodes_of_interest = nodes_of_interest.sort_values()

        # noi contains all property values
        noi = self.__graph.get_vertex_data(
            nodes_of_interest.values_host if self.is_mg
            else nodes_of_interest
        )
        noi_types = noi[self.__graph.type_col_name].cat.categories.values_host

        noi_index = {}
        noi_groups = {}
        noi_tensors = {}
        for t_code, t in enumerate(noi_types):
            noi_t = noi[noi[self.__graph.type_col_name].cat.codes == t_code]
            # noi_t should be sorted since the input nodes of interest were

            if len(noi_t) > 0:
                # store the renumbering for this vertex type
                # renumbered vertex id is the index of the old id
                noi_index[t] = (
                    noi_t[self.__graph.vertex_col_name].compute().to_cupy()
                    if self.is_mg
                    else noi_t[self.__graph.vertex_col_name].to_cupy()
                )

                # renumber for each noi group

                noi_groups[t] = self.from_dlpack(
                    cupy.arange(len(noi_t)).toDlpack()
                )

                # store the property data
                attrs = self._tensor_attr_dict[t]
                noi_tensors[t] = {
                    attr.attr_name: (
                        self.__get_tensor_from_dataframe(noi_t, attr)
                    )
                    for attr in attrs
                }

        return noi_index, noi_groups, noi_tensors

    def __get_renumbered_edges_from_sample(self, sampling_results, noi_index):
        eoi = self.__graph.get_edge_data(
            edge_ids=(
                sampling_results.indices.compute().values_host if self.is_mg
                else sampling_results.indices
            ),
            columns=[
                self.__graph.src_col_name,
                self.__graph.dst_col_name
            ]
        )
        eoi_types = eoi[self.__graph.type_col_name].cat.categories.values_host

        # PyG expects these to be pre-renumbered;
        # the pre-renumbering must match
        # the auto-renumbering
        row_dict = {}
        col_dict = {}
        for t_code, t in enumerate(eoi_types):
            t_pyg_type = self.__edge_types_to_attrs[t].edge_type
            src_type, edge_type, dst_type = t_pyg_type
            t_pyg_c_type = edge_type_to_str(t_pyg_type)

            eoi_t = eoi[eoi[self.__graph.type_col_name].cat.codes == t_code]

            if len(eoi_t) > 0:
                eoi_t = eoi_t.drop(self.__graph.edge_id_col_name, axis=1)

                sources = eoi_t[self.__graph.src_col_name]
                if self.is_mg:
                    sources = sources.compute()
                src_id_table = noi_index[src_type]

                src = self.from_dlpack(
                    cupy.searchsorted(
                        src_id_table,
                        sources.to_cupy()
                    ).toDlpack()
                )
                row_dict[t_pyg_c_type] = src

                destinations = eoi_t[self.__graph.dst_col_name]
                if self.is_mg:
                    destinations = destinations.compute()
                dst_id_table = noi_index[dst_type]

                dst = self.from_dlpack(
                    cupy.searchsorted(
                        dst_id_table, destinations.to_cupy()
                    ).toDlpack()
                )
                col_dict[t_pyg_c_type] = dst

        return row_dict, col_dict

    def put_tensor(self, tensor, attr):
        raise NotImplementedError('Adding properties not supported.')

    def create_named_tensor(self, attr_name, properties, vertex_type, dtype):
        """
            Create a named tensor that contains a subset of
            properties in the graph.
        """
        self._tensor_attr_dict[vertex_type].append(
            CuGraphTensorAttr(
                vertex_type,
                attr_name,
                properties=properties,
                dtype=dtype
            )
        )

    def __infer_x_and_y_tensors(self):
        """
        Infers the x and y default tensor attributes/features.
        """
        for vtype in self.__graph.vertex_types:
            df = self.__graph.get_vertex_data(types=[vtype])
            for rk in self.__reserved_keys:
                df = df.drop(rk, axis=1)

            if 'y' in df.columns:
                if df.y.isnull().values.any():
                    print(
                        f'Skipping definition of feature y'
                        f' for type {vtype} (null encountered)'
                    )
                else:
                    self.create_named_tensor(
                        'y',
                        ['y'],
                        vtype,
                        self.vertex_dtype
                    )
                df.drop('y', axis=1, inplace=True)

            x_cols = []
            for col in df.columns:
                if not df[col].isnull().values.any():
                    x_cols.append(col)

            if len(x_cols) == 0:
                print(
                        f'Skipping definition of feature'
                        f' x for type {vtype}'
                        f' (null encountered for all properties)'
                )
            else:
                self.create_named_tensor(
                        'x',
                        x_cols,
                        vtype,
                        self.property_dtype
                )

    def get_all_tensor_attrs(self):
        r"""Obtains all tensor attributes stored in this feature store."""
        # unpack and return the list of lists
        it = chain.from_iterable(self._tensor_attr_dict.values())
        return [CuGraphTensorAttr.cast(c) for c in it]

    def __get_tensor_from_dataframe(self, df, attr):
        df = df[attr.properties]

        if self.is_mg:
            df = df.compute()

        # FIXME handle vertices without properties
        output = self.from_dlpack(
            df.to_dlpack()
        )

        # FIXME look up the dtypes for x and other properties
        if output.dtype != attr.dtype:
            if self.__backend == 'torch':
                output = output.to(self.property_dtype)
            elif self.__backend == 'cupy':
                output = output.astype(self.property_dtype)
            else:
                raise ValueError(f'invalid backend {self.__backend}')

        return output

    def _get_tensor(self, attr):
        if attr.attr_name == 'x':
            cols = None
        else:
            cols = attr.properties

        idx = attr.index
        if self.__backend == 'torch' and not idx.is_cuda:
            idx = idx.cuda()
        idx = cupy.from_dlpack(idx.__dlpack__())

        if len(self.__graph.vertex_types) == 1:
            # make sure we don't waste computation if there's only 1 type
            df = self.__graph.get_vertex_data(
                vertex_ids=idx.get(),
                types=None,
                columns=cols
            )
        else:
            df = self.__graph.get_vertex_data(
                vertex_ids=idx.get(),
                types=[attr.group_name],
                columns=cols
            )

        return self.__get_tensor_from_dataframe(df, attr)

    def _multi_get_tensor(self, attrs):
        return [self._get_tensor(attr) for attr in attrs]

    def multi_get_tensor(self, attrs):
        r"""Synchronously obtains a :class:`FeatureTensorType` object from the
        feature store for each tensor associated with the attributes in
        `attrs`.

        Args:
            attrs (List[TensorAttr]): a list of :class:`TensorAttr` attributes
                that identify the tensors to get.

        Returns:
            List[FeatureTensorType]: a Tensor of the same type as the index for
                each attribute.

        Raises:
            KeyError: if a tensor corresponding to an attr was not found.
            ValueError: if any input `TensorAttr` is not fully specified.
        """
        attrs = [self._infer_unspecified_attr(self._tensor_attr_cls.cast(attr))
                 for attr in attrs]
        bad_attrs = [attr for attr in attrs if not attr.is_fully_specified()]
        if len(bad_attrs) > 0:
            raise ValueError(
                f"The input TensorAttr(s) '{bad_attrs}' are not fully "
                f"specified. Please fully specify them by specifying all "
                f"'UNSET' fields")

        tensors = self._multi_get_tensor(attrs)

        bad_attrs = [attrs[i] for i, v in enumerate(tensors) if v is None]
        if len(bad_attrs) > 0:
            raise KeyError(f"Tensors corresponding to attributes "
                           f"'{bad_attrs}' were not found")

        return [
            tensor
            for attr, tensor in zip(attrs, tensors)
        ]

    def get_tensor(self, *args, **kwargs):
        r"""Synchronously obtains a :class:`FeatureTensorType` object from the
        feature store. Feature store implementors guarantee that the call
        :obj:`get_tensor(put_tensor(tensor, attr), attr) = tensor` holds.

        Args:
            **attr (TensorAttr): Any relevant tensor attributes that correspond
                to the feature tensor. See the :class:`TensorAttr`
                documentation for required and optional attributes. It is the
                job of implementations of a :class:`FeatureStore` to store this
                metadata in a meaningful way that allows for tensor retrieval
                from a :class:`TensorAttr` object.

        Returns:
            FeatureTensorType: a Tensor of the same type as the index.

        Raises:
            KeyError: if the tensor corresponding to attr was not found.
            ValueError: if the input `TensorAttr` is not fully specified.
        """

        attr = self._tensor_attr_cls.cast(*args, **kwargs)
        attr = self._infer_unspecified_attr(attr)

        if not attr.is_fully_specified():
            raise ValueError(f"The input TensorAttr '{attr}' is not fully "
                             f"specified. Please fully specify the input by "
                             f"specifying all 'UNSET' fields.")

        tensor = self._get_tensor(attr)
        if tensor is None:
            raise KeyError(f"A tensor corresponding to '{attr}' was not found")
        return tensor

    def _get_tensor_size(self, attr):
        return self._get_tensor(attr).size

    def get_tensor_size(self, *args, **kwargs):
        r"""Obtains the size of a tensor given its attributes, or :obj:`None`
        if the tensor does not exist."""
        attr = self._tensor_attr_cls.cast(*args, **kwargs)
        if not attr.is_set('index'):
            attr.index = None
        return self._get_tensor_size(attr)

    def _remove_tensor(self, attr):
        raise NotImplementedError('Removing features not supported')

    def _infer_unspecified_attr(self, attr):
        if attr.properties == _field_status.UNSET:
            # attempt to infer property names
            if attr.group_name in self._tensor_attr_dict:
                for n in self._tensor_attr_dict[attr.group_name]:
                    if attr.attr_name == n.attr_name:
                        attr.properties = n.properties
            else:
                raise KeyError(f'Invalid group name {attr.group_name}')

        if attr.dtype == _field_status.UNSET:
            # attempt to infer dtype
            if attr.group_name in self._tensor_attr_dict:
                for n in self._tensor_attr_dict[attr.group_name]:
                    if attr.attr_name == n.attr_name:
                        attr.dtype = n.dtype

        return attr

    def __len__(self):
        return len(self.get_all_tensor_attrs())


def edge_type_to_str(edge_type):
    """
    Converts the PyG (src, type, dst) edge representation into
    the equivalent C++ representation.

    edge_type : The PyG (src, type, dst) tuple edge representation
        to convert to the C++ representation.
    """
    # Since C++ cannot take dictionaries with tuples as key as input, edge type
    # triplets need to be converted into single strings.
    return edge_type if isinstance(edge_type, str) else '__'.join(edge_type)
