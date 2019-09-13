Graph Partitioning (parla.partitioning)
=======================================

.. automodule:: parla.partitioning
   :no-members:

Partition
---------

.. autoclass:: Partition
   :no-members:
   :members: edges, vertex_global_ids, vertex_masters, edge_masters

Partitioning Algorithm
----------------------

.. autoclass:: PartitioningAlgorithm
   :no-members:
   :members: graph_properties, vertex_masters, n_partitions, neighborhood_size, get_vertex_master, get_edge_master, partition

.. autoclass:: GraphProperties
   :members:
