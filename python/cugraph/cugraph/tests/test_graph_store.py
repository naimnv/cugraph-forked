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

from collections import defaultdict
import pytest
import cugraph
from cugraph.experimental import PropertyGraph
import numpy as np
import cudf
import cupy as cp
from cugraph.gnn import CuGraphStore
from cugraph.experimental.datasets import DATASETS


@pytest.mark.parametrize("graph_file", DATASETS)
def test_no_graph(graph_file):
    with pytest.raises(TypeError):
        gstore = cugraph.gnn.CuGraphStore()
        gstore.num_edges()


@pytest.mark.parametrize("graph_file", DATASETS)
def test_using_graph(graph_file):
    with pytest.raises(ValueError):
        g = graph_file.get_graph()
        cugraph.gnn.CuGraphStore(graph=g)


@pytest.mark.parametrize("graph_file", DATASETS)
def test_using_pgraph(graph_file):
    g = graph_file.get_graph(create_using=cugraph.Graph(directed=True))
    cu_M = graph_file.get_edgelist().rename(
        columns={"src": "0", "dst": "1", "wgt": "2"})
    pG = PropertyGraph()
    pG.add_edge_data(cu_M, vertex_col_names=("0", "1"), property_columns=None)

    gstore = cugraph.gnn.CuGraphStore(graph=pG)

    assert g.number_of_edges() == pG.get_num_edges()
    assert g.number_of_edges() == gstore.num_edges()
    assert g.number_of_vertices() == pG.get_num_vertices()
    assert g.number_of_vertices() == gstore.num_vertices


@pytest.mark.parametrize("graph_file", DATASETS)
def test_node_data_pg(graph_file):
    cu_M = graph_file.get_edgelist().rename(
        columns={"src": "0", "dst": "1", "wgt": "2"})
    pG = PropertyGraph()
    gstore = cugraph.gnn.CuGraphStore(graph=pG, backend_lib="cupy")
    gstore.add_edge_data(
        cu_M, vertex_col_names=("0", "1"), feat_name="edge_feat"
    )

    edata_f = gstore.get_edge_storage("edge_feat")
    edata = edata_f.fetch(indices=[0, 1], device="cuda")

    assert edata.shape[0] > 0


@pytest.mark.skip("Skipping egonet testing for now")
@pytest.mark.parametrize("graph_file", DATASETS)
def test_egonet(graph_file):

    from cugraph.community.egonet import batched_ego_graphs

    g = graph_file.get_graph(create_using=cugraph.Graph(directed=True))
    cu_M = graph_file.get_edgelist().rename(
        columns={"src": "0", "dst": "1", "wgt": "2"})
    pG = PropertyGraph()
    gstore = cugraph.gnn.CuGraphStore(graph=pG, backend_lib="cupy")
    gstore.add_edge_data(
        cu_M, vertex_col_names=("0", "1"), feat_name="edge_feat"
    )

    nodes = [1, 2]

    ego_edge_list1, seeds_offsets1 = gstore.egonet(nodes, k=1)
    ego_edge_list2, seeds_offsets2 = batched_ego_graphs(g, nodes, radius=1)

    assert ego_edge_list1 == ego_edge_list2
    assert seeds_offsets1 == seeds_offsets2


@pytest.mark.skip("Skipping egonet testing for now")
@pytest.mark.parametrize("graph_file", DATASETS)
def test_workflow(graph_file):
    # from cugraph.community.egonet import batched_ego_graphs
    cu_M = graph_file.get_edgelist().rename(
        columns={"src": "0", "dst": "1", "wgt": "2"})
    pg = PropertyGraph()
    gstore = cugraph.gnn.CuGraphStore(graph=pg)
    gstore.add_edge_data(cu_M, vertex_col_names=("0", "1"), feat_name="feat")
    nodes = gstore.get_vertex_ids()
    num_nodes = len(nodes)

    assert num_nodes > 0

    sampled_nodes = nodes[:5]

    ego_edge_list, seeds_offsets = gstore.egonet(sampled_nodes, k=1)

    assert len(ego_edge_list) > 0


@pytest.mark.parametrize("graph_file", DATASETS)
def test_sample_neighbors(graph_file):
    cu_M = graph_file.get_edgelist().rename(
        columns={"src": "0", "dst": "1", "wgt": "2"})
    pg = PropertyGraph()
    gstore = cugraph.gnn.CuGraphStore(graph=pg)
    gstore.add_edge_data(cu_M, feat_name="feat", vertex_col_names=("0", "1"))

    nodes = gstore.get_vertex_ids()
    num_nodes = len(nodes)

    assert num_nodes > 0

    sampled_nodes = nodes[:5].to_dlpack()

    parents_cap, children_cap, edge_id_cap = gstore.sample_neighbors(
        sampled_nodes, 2
    )

    parents_list = cudf.from_dlpack(parents_cap)
    assert len(parents_list) > 0


@pytest.mark.parametrize("graph_file", DATASETS)
def test_sample_neighbor_neg_one_fanout(graph_file):
    cu_M = graph_file.get_edgelist().rename(
        columns={"src": "0", "dst": "1", "wgt": "2"})
    pg = PropertyGraph()
    gstore = cugraph.gnn.CuGraphStore(graph=pg)
    gstore.add_edge_data(cu_M, feat_name="edge_k", vertex_col_names=("0", "1"))

    nodes = gstore.get_vertex_ids()
    sampled_nodes = nodes[:5].to_dlpack()
    # -1, default fan_out
    parents_cap, children_cap, edge_id_cap = gstore.sample_neighbors(
        sampled_nodes, -1
    )
    parents_list = cudf.from_dlpack(parents_cap)
    assert len(parents_list) > 0


@pytest.mark.parametrize("graph_file", DATASETS)
def test_get_node_storage_graph_file(graph_file):
    cu_M = graph_file.get_edgelist().rename(
        columns={"src": "0", "dst": "1", "wgt": "2"})

    pg = PropertyGraph()
    gstore = cugraph.gnn.CuGraphStore(graph=pg, backend_lib="cupy")

    gstore.add_edge_data(
        cu_M,
        feat_name="feat",
        vertex_col_names=("0", "1"),
    )

    num_nodes = gstore.num_nodes()
    df_feat = cudf.DataFrame()
    df_feat["node_id"] = np.arange(num_nodes)
    df_feat["val0"] = [float(i + 1) for i in range(num_nodes)]
    df_feat["val1"] = [float(i + 2) for i in range(num_nodes)]
    gstore.add_node_data(
        df_feat,
        feat_name="node_feat",
        node_col_name="node_id",
    )

    ndata_f = gstore.get_node_storage(feat_name="node_feat")
    ndata = ndata_f.fetch([0, 1, 2], device="cuda")

    assert ndata.shape[0] > 0


@pytest.mark.parametrize("graph_file", DATASETS)
def test_edge_storage_data_graph_file(graph_file):
    cu_M = graph_file.get_edgelist().rename(
        columns={"src": "0", "dst": "1", "wgt": "2"})
    pg = PropertyGraph()
    gstore = cugraph.gnn.CuGraphStore(graph=pg, backend_lib="cupy")
    gstore.add_edge_data(cu_M, vertex_col_names=("0", "1"), feat_name="edge_k")

    edata_s = gstore.get_edge_storage(feat_name="edge_k")
    edata = edata_s.fetch([0, 1, 2, 3], device="cuda")
    assert edata.shape[0] > 0


dataset1 = {
    "merchants": [
        [
            "merchant_id",
            "merchant_locaton",
            "merchant_size",
            "merchant_sales",
            "merchant_num_employees",
        ],
        [
            (11, 78750, 44, 123.2, 12),
            (4, 78757, 112, 234.99, 18),
            (21, 44145, 83, 992.1, 27),
            (16, 47906, 92, 32.43, 5),
            (86, 47906, 192, 2.43, 51),
        ],
    ],
    "users": [
        ["user_id", "user_location", "vertical"],
        [
            (89021, 78757, 0),
            (32431, 78750, 1),
            (89216, 78757, 1),
            (78634, 47906, 0),
        ],
    ],
    "taxpayers": [
        # We assume unique ids
        # now for graphstore to match DGL
        # https://github.com/rapidsai/cugraph/pull/2697#issuecomment-1247442646
        ["payer_id", "amount"],
        [
            (110, 1123.98),
            (40, 3243.7),
            (210, 8932.3),
            (160, 3241.77),
            (860, 789.2),
            (890210, 23.98),
            (786340, 41.77),
        ],
    ],
    "transactions": [
        ["user_id", "merchant_id", "volume", "time", "card_num"],
        [
            (89021, 11, 33.2, 1639084966.5513437, 123456),
            (89216, 4, None, 1639085163.481217, 8832),
            (78634, 16, 72.0, 1639084912.567394, 4321),
            (32431, 4, 103.2, 1639084721.354346, 98124),
        ],
    ],
    "relationships": [
        ["user_id_1", "user_id_2", "relationship_type"],
        [
            (89216, 89021, 9),
            (89216, 32431, 9),
            (32431, 78634, 8),
            (78634, 89216, 8),
        ],
    ],
    "referrals": [
        ["user_id_1", "user_id_2", "merchant_id", "stars"],
        [
            (89216, 78634, 11, 5),
            (89021, 89216, 4, 4),
            (89021, 89216, 21, 3),
            (89021, 89216, 11, 3),
            (89021, 78634, 21, 4),
            (78634, 32431, 11, 4),
        ],
    ],
}


# util to create dataframe
def create_df_from_dataset(col_n, rows):
    data_d = defaultdict(list)
    for row in rows:
        for col_id, col_v in enumerate(row):
            data_d[col_n[col_id]].append(col_v)
    return cudf.DataFrame(data_d)


# @pytest.fixture()
# TODO: Creating a fixture seems to hang the pytests


def get_dataset1_CuGraphStore():
    merchant_df = create_df_from_dataset(
        dataset1["merchants"][0], dataset1["merchants"][1]
    )
    user_df = create_df_from_dataset(
        dataset1["users"][0], dataset1["users"][1]
    )
    taxpayers_df = create_df_from_dataset(
        dataset1["taxpayers"][0], dataset1["taxpayers"][1]
    )
    transactions_df = create_df_from_dataset(
        dataset1["transactions"][0], dataset1["transactions"][1]
    )
    relationships_df = create_df_from_dataset(
        dataset1["relationships"][0], dataset1["relationships"][1]
    )
    referrals_df = create_df_from_dataset(
        dataset1["referrals"][0], dataset1["referrals"][1]
    )

    pG = PropertyGraph()
    graph = CuGraphStore(pG, backend_lib="cupy")
    # Vertex and edge data is added as one or more DataFrames; either a Pandas
    # DataFrame to keep data on the CPU, a cuDF DataFrame to keep data on GPU,
    # or a dask_cudf DataFrame to keep data on distributed GPUs.

    # For dataset1: vertices are merchants and users, edges are transactions,
    # relationships, and referrals.

    # property_columns=None (the default) means all columns except
    # vertex_col_name will be used as properties for the vertices/edges.

    graph.add_node_data(merchant_df, "merchant_id", "merchant_k", "merchant")
    graph.add_node_data(user_df, "user_id", "user_k", "user")
    graph.add_node_data(taxpayers_df, "payer_id", "taxpayers_k", "taxpayers")

    graph.add_edge_data(
        referrals_df,
        ("user_id_1", "user_id_2"),
        "referrals_k",
        "referrals",
    )
    graph.add_edge_data(
        relationships_df,
        ("user_id_1", "user_id_2"),
        "relationships_k",
        "relationships",
    )
    graph.add_edge_data(
        transactions_df,
        ("user_id", "merchant_id"),
        "transactions_k",
        "transactions",
    )

    return graph


def test_num_nodes_gs():
    gs = get_dataset1_CuGraphStore()
    # Added unique id in tax_payer so changed to 16
    assert gs.num_nodes() == 16


def test_num_edges():
    gs = get_dataset1_CuGraphStore()
    assert gs.num_edges() == 14


def test_etypes():
    dataset1_CuGraphStore = get_dataset1_CuGraphStore()
    assert dataset1_CuGraphStore.etypes == [
        'referrals', 'relationships', 'transactions'
    ]


def test_ntypes():
    dataset1_CuGraphStore = get_dataset1_CuGraphStore()
    assert dataset1_CuGraphStore.ntypes == ['merchant', 'taxpayers', 'user']


def test_get_node_storage_gs():
    dataset1_CuGraphStore = get_dataset1_CuGraphStore()
    fs = dataset1_CuGraphStore.get_node_storage(
        feat_name="merchant_k", ntype="merchant"
    )
    # indices = [11, 4, 21, 316, 11]
    indices = [11, 4, 21, 316]

    merchant_gs = fs.fetch(indices, device="cuda")
    merchant_df = create_df_from_dataset(
        dataset1["merchants"][0], dataset1["merchants"][1]
    )
    cudf_ar = merchant_df.set_index("merchant_id").loc[indices].values
    assert cp.allclose(cudf_ar, merchant_gs)


def test_get_edge_storage_gs():
    dataset1_CuGraphStore = get_dataset1_CuGraphStore()
    fs = dataset1_CuGraphStore.get_edge_storage(
        "relationships_k", "relationships"
    )
    relationship_t = fs.fetch([6, 7, 8], device="cuda")

    relationships_df = create_df_from_dataset(
        dataset1["relationships"][0], dataset1["relationships"][1]
    )
    cudf_ar = relationships_df["relationship_type"].iloc[[0, 1, 2]].values

    assert cp.allclose(cudf_ar, relationship_t)


@pytest.mark.skip("Skipping because it tends to hang sometimes")
def test_sampling_gs_heterogeneous_ds1():
    node_d = {"merchant_id": cudf.Series([4], dtype="int64").to_dlpack()}
    gs = get_dataset1_CuGraphStore()
    sampled_obj = gs.sample_neighbors(node_d, fanout=1)
    sampled_d = convert_dlpack_to_cudf_ser(sampled_obj)
    # Ensure we get sample from at at least one of the etypes
    src_ser = cudf.concat([s for s, _, _ in sampled_d.values()])
    assert len(src_ser) != 0


@pytest.mark.skip("Skipping because it tends to hang sometimes")
def test_sampling_gs_heterogeneous_ds1_neg_one_fanout():

    node_d = {"merchant_id": cudf.Series([4], dtype="int64").to_dlpack()}
    gs = get_dataset1_CuGraphStore()
    sampled_obj = gs.sample_neighbors(node_d, fanout=-1)
    sampled_d = convert_dlpack_to_cudf_ser(sampled_obj)
    # Ensure we get sample from at at least one of the etypes
    src_ser = cudf.concat([s for s, _, _ in sampled_d.values()])
    assert len(src_ser) != 0


def test_sampling_homogeneous_gs_out_dir():
    src_ser = cudf.Series([1, 1, 1, 1, 1, 2, 2, 3])
    dst_ser = cudf.Series([2, 3, 4, 5, 6, 3, 4, 7])
    df = cudf.DataFrame(
        {"src": src_ser, "dst": dst_ser, "edge_id": np.arange(len(src_ser))}
    )
    pg = PropertyGraph()
    gs = CuGraphStore(pg)
    gs.add_edge_data(df, ["src", "dst"], feat_name="edges")

    # below are obtained from dgl runs on the same graph
    expected_out = {
        1: ([1, 1, 1, 1, 1], [2, 3, 4, 5, 6]),
        2: ([2, 2], [3, 4]),
        3: ([3], [7]),
        4: ([], []),
    }

    for seed in expected_out.keys():
        seed_cap = cudf.Series([seed]).to_dlpack()
        sample_src, sample_dst, sample_eid = gs.sample_neighbors(
            seed_cap, fanout=9, edge_dir="out"
        )
        if sample_src is None:
            sample_src = cudf.Series([]).astype(np.int64)
            sample_dst = cudf.Series([]).astype(np.int64)
            sample_eid = cudf.Series([]).astype(np.int64)
        else:
            sample_src = cudf.from_dlpack(sample_src)
            sample_dst = cudf.from_dlpack(sample_dst)
            sample_eid = cudf.from_dlpack(sample_eid)

        output_df = cudf.DataFrame({"src": sample_src, "dst": sample_dst})
        output_df = output_df.sort_values(by=["src", "dst"])
        output_df = output_df.reset_index(drop=True)

        expected_df = cudf.DataFrame(
            {"src": expected_out[seed][0], "dst": expected_out[seed][1]}
        ).astype(np.int64)
        cudf.testing.assert_frame_equal(output_df, expected_df)

        sample_edge_id_df = cudf.DataFrame(
            {"src": sample_src, "dst": sample_dst, "edge_id": sample_eid}
        )
        assert_correct_eids(df, sample_edge_id_df)


def test_sampling_homogeneous_gs_in_dir():
    src_ser = cudf.Series([1, 1, 1, 1, 1, 2, 2, 3])
    dst_ser = cudf.Series([2, 3, 4, 5, 6, 3, 4, 7])
    df = cudf.DataFrame(
        {"src": src_ser, "dst": dst_ser, "edge_id": np.arange(len(src_ser))}
    )
    pg = PropertyGraph()
    gs = CuGraphStore(pg)
    gs.add_edge_data(df, ["src", "dst"], feat_name="edges")

    # below are obtained from dgl runs on the same graph
    expected_in = {
        1: ([], []),
        2: ([1], [2]),
        3: ([1, 2], [3, 3]),
        4: ([1, 2], [4, 4]),
    }

    for seed in expected_in.keys():
        seed_cap = cudf.Series([seed]).to_dlpack()
        sample_src, sample_dst, sample_eid = gs.sample_neighbors(
            seed_cap, fanout=9, edge_dir="in"
        )
        if sample_src is None:
            sample_src = cudf.Series([]).astype(np.int64)
            sample_dst = cudf.Series([]).astype(np.int64)
            sample_eid = cudf.Series([]).astype(np.int64)
        else:
            sample_src = cudf.from_dlpack(sample_src)
            sample_dst = cudf.from_dlpack(sample_dst)
            sample_eid = cudf.from_dlpack(sample_eid)

        output_df = cudf.DataFrame({"src": sample_src, "dst": sample_dst})
        output_df = output_df.sort_values(by=["src", "dst"])
        output_df = output_df.reset_index(drop=True)

        expected_df = cudf.DataFrame(
            {"src": expected_in[seed][0], "dst": expected_in[seed][1]}
        ).astype(np.int64)
        cudf.testing.assert_frame_equal(output_df, expected_df)

        sample_edge_id_df = cudf.DataFrame(
            {"src": sample_src, "dst": sample_dst, "edge_id": sample_eid}
        )

        assert_correct_eids(df, sample_edge_id_df)


def create_gs_heterogeneous_dgl_eg():
    pg = PropertyGraph()
    gs = CuGraphStore(pg)

    # Add Edge Data
    src_ser = [0, 1, 2, 0, 1, 2, 7, 9, 10, 11]
    dst_ser = [3, 4, 5, 6, 7, 8, 6, 6, 6, 6]
    etype_ser = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    edge_feat = [10, 10, 10, 11, 11, 11, 12, 12, 12, 13]

    etype_map = {
        0: "('nt.a', 'connects', 'nt.b')",
        1: "('nt.a', 'connects', 'nt.c')",
        2: "('nt.c', 'connects', 'nt.c')",
    }

    df = cudf.DataFrame(
        {
            "src": src_ser,
            "dst": dst_ser,
            "etype": etype_ser,
            "edge_feat": edge_feat,
        }
    )
    for e in df["etype"].unique().values_host:
        subset_df = df[df["etype"] == e][["src", "dst", "edge_feat"]]
        gs.add_edge_data(
            subset_df,
            ["src", "dst"],
            feat_name="edge_feat",
            etype=etype_map[e],
        )

    # Add Node Data
    node_ser = np.arange(0, 12)
    node_type = ["nt.a"] * 3 + ["nt.b"] * 3 + ["nt.c"] * 6
    node_feat = np.arange(0, 12) * 10
    df = cudf.DataFrame(
        {"node_id": node_ser, "ntype": node_type, "node_feat": node_feat}
    )
    for n in df["ntype"].unique().values_host:
        subset_df = df[df["ntype"] == n][["node_id", "node_feat"]]
        gs.add_node_data(
            subset_df, "node_id", feat_name="node_feat", ntype=str(n)
        )

    return gs


@pytest.mark.skip("Skipping because tends to hang")
def test_sampling_gs_heterogeneous_in_dir():
    gs = create_gs_heterogeneous_dgl_eg()
    # DGL expected_output from
    # https://gist.github.com/VibhuJawa/f85fda8e1183886078f2a34c28c4638c
    expeced_val_d = {
        6: {
            "('nt.a', 'connects', 'nt.b')": (
                cudf.Series([], dtype=np.int32),
                cudf.Series([], dtype=np.int32),
            ),
            "('nt.a', 'connects', 'nt.c')": (
                cudf.Series([0]),
                cudf.Series([6]),
            ),
            "('nt.c', 'connects', 'nt.c')": (
                cudf.Series([7, 9, 10, 11]),
                cudf.Series([6, 6, 6, 6]),
            ),
        },
        7: {
            "('nt.a', 'connects', 'nt.b')": (
                cudf.Series([], dtype=np.int32),
                cudf.Series([], dtype=np.int32),
            ),
            "('nt.a', 'connects', 'nt.c')": (
                cudf.Series([1]),
                cudf.Series([7]),
            ),
            "('nt.c', 'connects', 'nt.c')": (
                cudf.Series([], dtype=np.int32),
                cudf.Series([], dtype=np.int32),
            ),
        },
    }

    for seed in expeced_val_d.keys():
        fanout = 4
        sampled_node_p = cudf.Series(seed).astype(np.int32).to_dlpack()
        sampled_g = gs.sample_neighbors(
            {"nt.c": sampled_node_p}, fanout=fanout, edge_dir="in"
        )
        sampled_g = convert_dlpack_dict_to_df(sampled_g)
        for etype, df in sampled_g.items():
            output_df = (
                df[["src", "dst"]]
                .sort_values(by=["src", "dst"])
                .reset_index(drop=True)
                .astype(np.int32)
            )
            expected_df = cudf.DataFrame(
                {
                    "src": expeced_val_d[seed][etype][0],
                    "dst": expeced_val_d[seed][etype][1],
                }
            ).astype(np.int32)
            cudf.testing.assert_frame_equal(output_df, expected_df)


@pytest.mark.skip("Skipping because tends to hang")
def test_sampling_gs_heterogeneous_out_dir():
    gs = create_gs_heterogeneous_dgl_eg()
    # DGL expected_output from
    # https://gist.github.com/VibhuJawa/f85fda8e1183886078f2a34c28c4638c
    expeced_val_d = {
        0: {
            "('nt.a', 'connects', 'nt.b')": (
                cudf.Series([0], dtype=np.int32),
                cudf.Series([3], dtype=np.int32),
            ),
            "('nt.a', 'connects', 'nt.c')": (
                cudf.Series([0]),
                cudf.Series([6]),
            ),
            "('nt.c', 'connects', 'nt.c')": (cudf.Series([]), cudf.Series([])),
        },
        1: {
            "('nt.a', 'connects', 'nt.b')": (
                cudf.Series([1], dtype=np.int32),
                cudf.Series([4], dtype=np.int32),
            ),
            "('nt.a', 'connects', 'nt.c')": (
                cudf.Series([1]),
                cudf.Series([7]),
            ),
            "('nt.c', 'connects', 'nt.c')": (
                cudf.Series([], dtype=np.int32),
                cudf.Series([], dtype=np.int32),
            ),
        },
        2: {
            "('nt.a', 'connects', 'nt.b')": (
                cudf.Series([2], dtype=np.int32),
                cudf.Series([5], dtype=np.int32),
            ),
            "('nt.a', 'connects', 'nt.c')": (
                cudf.Series([2]),
                cudf.Series([8]),
            ),
            "('nt.c', 'connects', 'nt.c')": (
                cudf.Series([], dtype=np.int32),
                cudf.Series([], dtype=np.int32),
            ),
        },
    }

    for seed in expeced_val_d.keys():
        fanout = 4
        sampled_node_p = cudf.Series(seed).astype(np.int32).to_dlpack()
        sampled_g = gs.sample_neighbors(
            {"nt.a": sampled_node_p}, fanout=fanout, edge_dir="out"
        )
        sampled_g = convert_dlpack_dict_to_df(sampled_g)
        for etype, df in sampled_g.items():
            output_df = (
                df[["src", "dst"]]
                .sort_values(by=["src", "dst"])
                .reset_index(drop=True)
                .astype(np.int32)
            )
            expected_df = cudf.DataFrame(
                {
                    "src": expeced_val_d[seed][etype][0],
                    "dst": expeced_val_d[seed][etype][1],
                }
            ).astype(np.int32)
            cudf.testing.assert_frame_equal(output_df, expected_df)


@pytest.mark.skip("Skipping because tends to hang")
def test_sampling_dgl_heterogeneous_gs_m_fanouts():
    gs = create_gs_heterogeneous_dgl_eg()
    # Test against DGLs output
    # See below notebook
    # https://gist.github.com/VibhuJawa/f85fda8e1183886078f2a34c28c4638c

    expected_output = {
        1: {
            "('nt.a', 'connects', 'nt.b')": 0,
            "('nt.a', 'connects', 'nt.c')": 1,
            "('nt.c', 'connects', 'nt.c')": 1,
        },
        2: {
            "('nt.a', 'connects', 'nt.b')": 0,
            "('nt.a', 'connects', 'nt.c')": 1,
            "('nt.c', 'connects', 'nt.c')": 2,
        },
        3: {
            "('nt.a', 'connects', 'nt.b')": 0,
            "('nt.a', 'connects', 'nt.c')": 1,
            "('nt.c', 'connects', 'nt.c')": 3,
        },
        -1: {
            "('nt.a', 'connects', 'nt.b')": 0,
            "('nt.a', 'connects', 'nt.c')": 1,
            "('nt.c', 'connects', 'nt.c')": 4,
        },
    }

    for fanout in [1, 2, 3, -1]:
        sampled_node = [6]
        sampled_node_p = cudf.Series(sampled_node).to_dlpack()
        sampled_g = gs.sample_neighbors(
            {"nt.c": sampled_node_p}, fanout=fanout
        )
        sampled_g = convert_dlpack_dict_to_df(sampled_g)
        for etype, output_df in sampled_g.items():
            assert expected_output[fanout][etype] == len(output_df)


def assert_correct_eids(edge_df, sample_edge_id_df):
    # We test that all src, dst correspond to the correct
    # eids in the sample_edge_id_df
    # we do this by ensuring that the inner merge to edge_df
    # remains the same as sample_edge_id_df
    # if they don't correspond correctly
    # the inner merge would fail

    sample_edge_id_df = sample_edge_id_df.sort_values(by="edge_id")
    sample_edge_id_df = sample_edge_id_df.reset_index(drop=True)

    sample_merged_df = sample_edge_id_df.merge(edge_df, how="inner")
    sample_merged_df = sample_merged_df.sort_values(by="edge_id")
    sample_merged_df = sample_merged_df.reset_index(drop=True)
    assert sample_merged_df.equals(sample_edge_id_df)


def convert_dlpack_to_cudf_ser(cap_d):
    ser_d = {}
    for etype, (s, d, eid) in cap_d.items():
        if s is None:
            sample_src = cudf.Series([]).astype(np.int64)
            sample_dst = cudf.Series([]).astype(np.int64)
            sample_eid = cudf.Series([]).astype(np.int64)
        else:
            sample_src = cudf.from_dlpack(s)
            sample_dst = cudf.from_dlpack(d)
            sample_eid = cudf.from_dlpack(eid)

        ser_d[etype] = (sample_src, sample_dst, sample_eid)

    return ser_d


def convert_dlpack_dict_to_df(d):
    df_d = convert_dlpack_to_cudf_ser(d)
    df_d = {
        k: cudf.DataFrame({"src": s, "dst": d, "eids": e})
        for k, (s, d, e) in df_d.items()
    }
    return df_d
