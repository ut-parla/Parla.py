import pytest

from parla import Parla, spawn, TaskSpace, parray
from parla.cuda import gpu
from parla.cpu import cpu
from parla.parray.coherence import Coherence
import numpy as np

def test_parray_creation():
    with Parla():
        A = parray.asarray([[1, 2], [3, 4]])

        a = A[0]
        assert A[0,1] == 2
        assert A[1,0] == 3
        assert A[1,1] == 4
        assert np.array_equal(A, np.asarray([[1, 2], [3, 4]]))

def test_parray_task():
    with Parla():
        @spawn(placement=cpu)
        def test():
            np.random.seed(10)
            # Construct input data
            a = np.array([[1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6]])
            b = np.array([[1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6]])
            a = parray.asarray(a, name="A")
            b = parray.asarray(b, name="B")

            ts = TaskSpace("CopyBack")

            assert a._current_device_index == -1
            assert a._array._buffer[1] is None
            assert a._array._buffer[-1] is not None
            assert a._coherence._local_states[1] == Coherence.INVALID
            assert a._coherence._local_states[-1] == Coherence.MODIFIED

            @spawn(ts[1], placement=gpu[1], output=[b])
            def check_array_write():
                assert b[0,0] == 1
                assert b._current_device_index == 1
                b.print_overview()
                
                b[1,1] = 0
                assert b[1,1] == 0
                assert b._array._buffer[1] is not None
                assert b._array._buffer[-1] is None
                assert b._coherence._local_states[-1] == Coherence.INVALID
                assert b._coherence._local_states[1] == Coherence.MODIFIED

            @spawn(ts[2], dependencies=[ts[1]], placement=gpu[0], output=[a[0:2], a[2]])
            def check_array_slicing():
                assert a[1,0] == 1
                assert a._current_device_index == 0
                a[0:2].print_overview()
                
                a[1,1] = 0
                assert a[1,1] == 0
                assert a._array._buffer[-1] is not None
                assert isinstance(a._array._buffer[0], list)
                assert a._coherence._local_states[-1] == Coherence.INVALID
                assert isinstance(a._coherence._local_states[0], dict)

            @spawn(ts[3], dependencies=[ts[2]], placement=cpu, output=[a])
            def check_array_write_back():
                assert a[1,1] == 0
                assert a._current_device_index == -1
                a.print_overview()
                
                assert a._array._buffer[-1] is not None
                assert a._array._buffer[0] is None
                assert a._coherence._local_states[-1] == Coherence.MODIFIED
                assert a._coherence._local_states[0] == Coherence.INVALID

            @spawn(ts[4], dependencies=[ts[3]], placement=gpu[1], inout=[a])
            def check_array_update():
                a.update(np.array([1,2,3,4]))  # here np array are converted to cupy array

                assert len(a) == 4
                assert a[-1] == 4
                assert a._coherence.owner == 1

            @spawn(ts[5], dependencies=[ts[4]], placement=gpu[0], inout=[a])
            def check_array_indexing():
                a[np.array([0,2])] = 0
                assert len(a) == 4
                assert a[0] == 0
                assert a._coherence.owner == 0

if __name__=="__main__":
    test_parray_creation()
    test_parray_task()