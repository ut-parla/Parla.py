"""
This module implement dataflow tracking.
It is used to provide data-aware scheduling,
and also eager-fashion automatic data movement.
"""

from typing import Collection, Any
from itertools import chain

from parla.task_runtime import get_current_devices
from parla.cpu_impl import cpu
from parla.parray.coherence import CPU_INDEX

class DataflowIterator:
    """
    Itrator class for Dataflow.
    """
    def __init__(self, df):
        self._df = df
        self._idx = 0

    def __next__(self):
        """
        Return the next value from Dataflow's data lists:
        input -> output -> in/output lists
        """
        if self._idx < (len(self._df._input) + len(self._df._output) +
                        len(self._df._inout)):
            if self._idx < len(self._df._input):
                # First, iterate input data operands.
                cur_item = self._df._input[self._idx]
            elif self._idx < (len(self._df._input) + len(self._df._output)):
                # Second, iterate output data operands.
                cur_item = self._df._output[self._idx - len(self._df._input)]
            else:
                # Third, iterate input/output data operands.
                cur_item = self._df._inout[self._idx - len(self._df._input) -
                                           len(self._df._output)]
            self._idx += 1
            return cur_item
        raise StopIteration


class Dataflow:
    """
    The data reference of input/output/inout of a task
    """

    def __init__(self, input: Collection[Any], output: Collection[Any], inout: Collection[Any]):
        self._input = input
        self._output = output
        self._inout = inout

    def auto_move(self):
        """
        Move all data to the current device (of the corresponding tasks).
        Only PArray is supported.
        """
        # query the current device id
        device = get_current_devices()[0]
        if device.architecture == cpu:
            device_id = CPU_INDEX
        else:  # arch is GPU
            device_id = device.index

        for array in self._input:
            array._auto_move(device_id, do_write=False)

        for array in chain(self._output, self._inout):
            array._auto_move(device_id, do_write=True)

    def __iter__(self):
        return DataflowIterator(self)

