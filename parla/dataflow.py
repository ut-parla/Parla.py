"""
This module implement dataflow tracking.
It is used to provide data-aware scheduling,
and also eager-fashion automatic data movement.
"""

from typing import List, Any
from itertools import chain

from parla import task_runtime
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

    def __init__(self, input: List[Any], output: List[Any], inout: List[Any]):
        self._input = input
        self._output = output
        self._inout = inout

    @property
    def input(self) -> List:
        if self._input == None:
            return []
        return self._input

    @property
    def output(self) -> List:
        if self._output == None:
            return []
        return self._output

    @property
    def inout(self) -> List:
        if self._inout == None:
            return []
        return self._inout

    def auto_move(self):
        """
        Move all data to the current device (of the corresponding tasks).
        Only PArray is supported.
        """
        # query the current device id
        device = task_runtime.get_current_devices()[0]
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

