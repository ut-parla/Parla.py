from core cimport *

cdef class PyGraph:
    cdef Graph* underlying

def py_distance(unsigned long self, int node, int slot):
    g = <Graph*>self
    cdef int c_node = node
    cdef int c_slot = slot
    with nogil:
        x =  read_distance(g, c_node, c_slot)
    return x

def py_bfs(unsigned long self, int source, int slot):
    g = <Graph*>self
    cdef int c_source = source
    cdef int c_slot = slot
    with nogil:
        bfs(g, c_source, c_slot)

def py_load_file(str fn):
    cdef bytes bytestring = fn.encode()
    cdef char *c_string = bytestring
    cdef unsigned long r
    with nogil:
        r = <unsigned long>load_file(c_string)
    return r

def py_init_galois(int nThreads):
    cdef int cnThreads = nThreads
    with nogil:
        init_galois(cnThreads)

def py_delete_galois():
    delete_galois()

def set_active_threads(int i):
    return setActiveThreads(i)

