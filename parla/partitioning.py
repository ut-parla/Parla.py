"""
This module implements graph/sparse matrix partitioning inspired by Gluon.
The goal of this module is to allow simple algorithms to be written against the current sequential implementation,
and eventually scale (without changes) to parallel or even distributed partitioning.
As such, many functions guarantee much less than they currently provide
(e.g., `~PartitioningAlgorithm.get_edge_master` is only guaranteed to see it's source and destination in `~PartitioningAlgorithm.vertex_masters` even though the initial sequential implementation will actually provide all masters.)

The partitioner uses two functions getVertexMaster and getEdgeMaster similar to those used by Gluon, but also provides access to vertex attributes like position.
The `~PartitioningAlgorithm.get_vertex_master` function selects the master based on vertex attributes or more typical graph properties.
The `~PartitioningAlgorithm.get_edge_master` function selects the master based on edge properties and the masters selected for the endpoints.

The partitioner also takes a `~PartitioningAlgorithm.neighborhood_size` parameter which specifies how far away from each vertex proxies are needed.
Edge proxies are included for all edges between vertices present on each node (either as master or as a proxy).

This module will work just like normal Gluon if neighborhood size is 1.
For vertex position based partitioning, we can just assign the node masters based on position and set an appropriate neighborhood.
For your sweeps algorithm, set neighborhood size to 2 and assign masters as needed.
"""
from abc import abstractmethod, ABCMeta, abstractproperty
from collections import namedtuple
from typing import Sequence, Set

import numpy as np
import scipy.sparse

__all__ = [
    "VertexID",
    "PartitionID",
    "GraphProperties",
    "PartitioningAlgorithm"
]

VertexID = int
PartitionID = int


class GraphProperties:
    def __init__(self, A: scipy.sparse.spmatrix):
        """
        Compute the graph properties of `A`. This is called by the `PartitioningAlgorithm` framework.
        """
        assert A.shape[0] == A.shape[1], "Parla only support partitioning homogeneous graphs with square edge matrices."
        self.A = A
        """
        The edge matrix of the graph.
        """
        self.n_vertices = A.shape[0]
        """
        The number of vertices.
        """
        self.n_edges = A.count_nonzero()
        """
        The number of edges.
        """
        nonzeros = (A != 0)
        # TODO: There MUST be a better way to do this.
        self.in_degree = nonzeros.sum(0).A.flatten()
        """
        A dense array containing the in degree of each vertex.
        """
        self.out_degree = nonzeros.sum(1).A.flatten()
        """
        A dense array containing the out degree of each vertex.
        """


class Partition(namedtuple("Partition", ("edges", "vertex_global_ids", "vertex_masters", "edge_masters"))):
    """
    An instance of `Partition` contains all of the data available to a specific partition.
    """

    edges: scipy.sparse.spmatrix
    """
    A sparse matrix containing all edges in this partition (both master copies and proxies).
    """
    vertex_global_ids: np.ndarray
    """
    A dense array of the global IDs of each vertex which is available locally (as a master copy or a proxy).
    In other words, this array is a mapping from local ID to global ID for all vertices what exist locally.
    The global IDs are always in ascending order.
    """
    vertex_masters: np.ndarray
    """
    An array of the master partitions for every vertex.
    """
    edge_masters: scipy.sparse.spmatrix
    """
    A sparse matrix of the master partitions for all locally available edges.
    The structure of this sparse matrix is identical to `edges`.    
    """


class PartitioningAlgorithm(metaclass=ABCMeta):
    graph_properties: GraphProperties
    """
    The `GraphProperties` of the graph bring partitioned.
    What data is available in it depends on the context in which it is accessed.
    """

    vertex_masters: np.ndarray
    """
    The vertex masters that have already been assigned.
    This data structure is not sequential consistent.
    See `get_vertex_master` and `get_edge_master` for information about what elements are guaranteed to be up to date during those calls.
    """

    # ... user-defined state ...

    @abstractproperty
    def n_partitions(self) -> int:
        """
        :return: The number of partitions this partitioner will create. (All partition IDs must be 0 < id < `n_partitions`)
        """
        pass

    @abstractproperty
    def neighborhood_size(self) -> int:
        """
        :return: The number of neighboring proxy vertices to include in each partition.
            Must be >= 0. A value of 0 will result in no proxies at all.
        """
        pass

    @abstractmethod
    def get_vertex_master(self, vertex_id: VertexID) -> PartitionID:
        """
        Compute the master partition ID for a vertex.
        This function may use `graph_properties` and the metadata of the specific vertex.

        :param vertex_id: The global ID of the vertex.
        :return: The master partition ID for the vertex.
        """
        pass

    @abstractmethod
    def get_edge_master(self, src_id: VertexID, dst_id: VertexID) -> PartitionID:
        """
        Compute the master partition ID for the specified edge.
        This function may use `vertex_masters`, but the only elements guaranteed to be present are `src_id` and `dst_id`.
        This function may use `graph_properties` freely.

        :param src_id: The global ID of the source vertex
        :param dst_id: The global ID of the target vertex
        :return: The master partition ID for the edge.
        """
        pass

    def partition(self, A: scipy.sparse.spmatrix, edge_matrix_type=scipy.sparse.csr_matrix) -> Sequence[Partition]:
        """
        Partition `A`.
        This operation mutates `self` and hence is not thread-safe.
        Some implementation of this may be internally parallel.

        :param A: The complete sparse edge matrix.
        :param edge_matrix_type: The type of edge matrix to build for each partition.
            This is used for both `Partition.edges` and `Partition.edge_masters`.
        :return: A sequence of `Partition` objects in ID order.
        """

        n_parts = self.n_partitions
        neighborhood_size = self.neighborhood_size
        self.graph_properties = GraphProperties(A)
        self.vertex_masters = np.empty(shape=(self.graph_properties.n_vertices,), dtype=int)
        self.vertex_masters[:] = -1
        edge_masters = scipy.sparse.csr_matrix(A.shape, dtype=int)
        partition_vertices: Sequence[Set[int]] = [set() for _ in range(n_parts)]
        # partition_n_edges = np.zeros(shape=(n_parts,), dtype=int)

        n, m = A.shape
        assert n == m, "Parla only support partitioning homogeneous graphs with square edge matrices."

        # Assign vertex masters
        for i in range(n):
            master = self.get_vertex_master(i)
            assert master >= 0 and master < n_parts, f"partition {master} is invalid ({n_parts} partitions)"
            self.vertex_masters[i] = master

        # TODO: This does not yet implement neighborhood > 1

        # Assign edge owners
        # TODO:PERFORMANCE: Iterate values without building index lists?
        for (i, j) in zip(*A.nonzero()):
            owner = self.get_edge_master(i, j)
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
            owner = self.get_edge_master(i, j)
            assert owner >= 0 and owner < n_parts, f"partition {owner} is invalid ({n_parts} partitions)"
            global_ids = partition_global_ids[owner]
            # TODO:PERFORANCE: Use a reverse index?
            partition_edges[owner][global_ids.searchsorted(i), global_ids.searchsorted(j)] = A[i, j]

        # Convert to compressed form
        return [Partition(edge_matrix_type(edges), global_ids, self.vertex_masters, edge_masters) for edges, global_ids
                in
                zip(partition_edges, partition_global_ids)]


