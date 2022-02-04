"""
This module implement dataflow tracking.
It is used to provide data-aware scheduling,
and also eager-fashion automatic data movement.
"""

from typing import Collection, Any
from itertools import chain

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
        for array in self._input:
            array._auto_move(do_write=False)

        for array in chain(self._output, self._inout):
            array._auto_move(do_write=True)