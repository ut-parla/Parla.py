"""
This module implement dataflow tracking.
It is used to provide data-aware scheduling,
and also eager-fashion automatic data movement.
"""

from typing import Collection, Any
from itertools import chain

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
        for array in chain(self._input, self._output, self._inout):
            array._auto_move()

    def __iter__(self):
        return DataflowIterator(self)
