from abc import abstractmethod, ABCMeta, abstractproperty
from typing import Mapping, MutableMapping, Sequence, Tuple

NodeID = int
PartitionID = int

class GraphProperties:
    def __init__(self, A):
        self.n_nodes = A.shape[0]
        self.n_edges = A.count_nonzero()
        self.in_degree = A.getnnz(0) # TODO
        self.out_degree = A.getnnz(1) # TODO

class PartitioningAlgorithm(metaclass=ABCMeta):
    graph_properties: GraphProperties
    node_masters: MutableMapping[NodeID, PartitionID]
    edge_owners: MutableMapping[Tuple[NodeID, NodeID], PartitionID]

    # ... user-defined state ...

    @abstractproperty
    def n_partitions(self) -> int:
        pass

    @abstractmethod
    def get_node_master(self, node_id: NodeID) -> PartitionID:
        pass

    @abstractmethod
    def get_edge_owner(self, src_id: NodeID, dst_id: NodeID) -> PartitionID:
        pass

    def partition(self, A) -> Sequence:
        self.graph_properties = GraphProperties(A)
        self.node_masters = dict() # TODO: Use array
        self.edge_owners = dict() # TODO: Use array?

        n, m = A.shape
        assert(n == m)

        # Assign mode masters
        for i in range(n):
            self.node_masters[i] = self.get_node_master(i)

        # Assign edge owners
        for i in range(n):
            for j in range(n):
                if A[i, j] != 0:
                    self.edge_owners[i, j] = self.get_edge_owner(i, j)

        # Partition matrix

        ## Build selector lists
        n_parts = self.n_partitions
        partition_selectors = [[] for x in range(n_parts)]
        for (i, j), partid in self.edge_owners.items():
            assert partid < n_parts
            # TODO

        ## Extract partition matricies
        out = []
        for indicies in partition_selectors:
            # TODO

        return out