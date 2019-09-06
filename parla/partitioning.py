from abc import abstractmethod, ABCMeta, abstractproperty
from collections import namedtuple
from typing import Sequence, Set

import numpy as np
import scipy.sparse

VertexID = int
PartitionID = int


class GraphProperties:
    def __init__(self, A: scipy.sparse.spmatrix):
        assert A.shape[0] == A.shape[1], "Parla only support partitioning homogeneous graphs with square edge matrices."
        self.n_vertices = A.shape[0]
        self.n_edges = A.count_nonzero()
        nonzeros = (A != 0)
        # TODO: There MUST be a better way to do this.
        self.in_degree = nonzeros.sum(0).A.flatten()
        self.out_degree = nonzeros.sum(1).A.flatten()


class Partition(namedtuple("Partition", ("edges", "vertex_global_ids", "vertex_masters"))):
    edges: scipy.sparse.spmatrix
    vertex_global_ids: np.ndarray
    vertex_masters: np.ndarray


class PartitioningAlgorithm(metaclass=ABCMeta):
    graph_properties: GraphProperties
    vertex_masters: np.ndarray

    # ... user-defined state ...

    @abstractproperty
    def n_partitions(self) -> int:
        pass

    @abstractmethod
    def get_vertex_master(self, vertex_id: VertexID) -> PartitionID:
        pass

    @abstractmethod
    def get_edge_owner(self, src_id: VertexID, dst_id: VertexID) -> PartitionID:
        pass

    def partition(self, A: scipy.sparse.spmatrix, edge_matrix_type=scipy.sparse.csr_matrix) -> Sequence[Partition]:
        n_parts = self.n_partitions
        self.graph_properties = GraphProperties(A)
        self.vertex_masters = np.empty(shape=(self.graph_properties.n_vertices,), dtype=int)
        self.vertex_masters[:] = -1
        partition_vertices: Sequence[Set[int]] = [set() for _ in range(n_parts)]
        # partition_n_edges = np.zeros(shape=(n_parts,), dtype=int)

        n, m = A.shape
        assert n == m, "Parla only support partitioning homogeneous graphs with square edge matrices."

        # Assign mode masters
        for i in range(n):
            master = self.get_vertex_master(i)
            assert master >= 0 and master < n_parts, f"partition {master} is invalid ({n_parts} partitions)"
            self.vertex_masters[i] = master

        # Assign edge owners
        # TODO:PERFORMANCE: Iterate values without building index lists?
        for (i, j) in zip(*A.nonzero()):
            owner = self.get_edge_owner(i, j)
            assert owner >= 0 and owner < n_parts, f"partition {owner} is invalid ({n_parts} partitions)"
            # partition_n_edges[owner] += 1
            partition_vertices[owner].add(i)
            partition_vertices[owner].add(j)

        # Build id maps
        partition_global_ids = [np.array(sorted(vs)) for vs in partition_vertices]

        # Construct in a efficiently updatable form (LiL)
        # TODO:PERFORMANCE: It would be more efficient to build directly in CSR or the appropriate output format.
        partition_edges = [scipy.sparse.lil_matrix((m.shape[0], m.shape[0])) for m in partition_global_ids]
        for (i, j) in zip(*A.nonzero()):
            owner = self.get_edge_owner(i, j)
            assert owner >= 0 and owner < n_parts, f"partition {owner} is invalid ({n_parts} partitions)"
            global_ids = partition_global_ids[owner]
            # TODO:PERFORANCE: Use a reverse index?
            partition_edges[owner][global_ids.searchsorted(i), global_ids.searchsorted(j)] = A[i, j]

        # Convert to compressed form
        return [Partition(edge_matrix_type(edges), global_ids, self.vertex_masters) for edges, global_ids in
                zip(partition_edges, partition_global_ids)]
