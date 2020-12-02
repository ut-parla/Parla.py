#distutils: language = c++

from libcpp.string cimport string

cdef extern from "galois/Galois.h" namespace "galois" nogil:
    unsigned int setActiveThreads(unsigned int)

cdef extern from "bfs_cpp.cpp" nogil:
    cppclass Graph:
        pass

    Graph* load_file(string& filename)
    void bfs(Graph* pGraph, int iSource, int slot)
    unsigned int read_distance(Graph* pGraph, int node, int slot)
    void init_galois()

