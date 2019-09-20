from math import floor, ceil

import numpy as np
import scipy.sparse as sp

from parla.partitioning import *


def test_trivial_partitioning():
    class ContiguousSource(PartitioningAlgorithm):
        @property
        def neighborhood_size(self) -> int:
            return 1

        def get_vertex_master(self, vertex_id: VertexID) -> PartitionID:
            block_size = ceil(self.graph_properties.n_vertices / self.n_partitions)
            return int(floor(vertex_id / block_size))

        def get_edge_master(self, src_id: VertexID, dst_id: VertexID) -> PartitionID:
            return self.vertex_masters[src_id]

        @property
        def n_partitions(self) -> int:
            return 2

    partitioner = ContiguousSource()
    A = sp.csr_matrix([
        [9, 1, 5, 0],
        [0, 2, 0, 6],
        [0, 3, 7, 0],
        [0, 4, 0, 8],
    ], dtype=int)
    A.eliminate_zeros()
    partitions = partitioner.partition(A)

    assert np.allclose(partitions[0].vertex_masters, np.array([0, 0, 1, 1]))
    assert np.allclose(partitions[0].edges.toarray(), np.array([
        [9, 1, 5, 0],
        [0, 2, 0, 6],
        [0, 0, 0, 0],
        [0, 0, 0, 0]]))
    assert partitions[0].edges.nnz == 5
    assert np.allclose(partitions[0].vertex_global_ids, np.array(range(4)))

    assert np.allclose(partitions[1].vertex_masters, np.array([0, 0, 1, 1]))
    assert np.allclose(partitions[1].edges.toarray(), np.array([
        [0, 0, 0],
        [3, 7, 0],
        [4, 0, 8]]))
    assert partitions[1].edges.nnz == 4
    assert np.allclose(partitions[1].vertex_global_ids, np.array(range(1, 4)))
