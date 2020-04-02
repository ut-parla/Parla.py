from typing import Collection, Union, Any, List

from .device import Architecture, Device
from .multiload import *
from .multiload import MultiloadContext
from .tasks import TaskID, Task, get_placement_for_set

__all__ = ["Context", "find_context"]

class Context:
    placement: List[Device]
    underlying: MultiloadContext

    def __init__(self, underlying: MultiloadContext, placement: Collection[Union[Architecture, Device, Task, TaskID, Any]] = None):
        self.placement = get_placement_for_set(placement)
        self.underlying = underlying
        assert getattr(underlying, "_high_level_context", None) is None, "Currently each MultiloadContext can only be associated with one Context."
        assert not find_context(self.placement, exact=True), "Having two contexts with identical placements is not allowed."
        underlying._high_level_context = self

        # TODO: Figure out how to factor this code out into some registry of setup functions.

        # Collect CPU numbers and set the affinity
        import parla.cpu
        cpus = []
        for d in self.placement:
            if d.architecture is parla.cpu.cpu:
                cpus.append(cpus)
        underlying.set_allowed_cpus(cpus)

        # TODO: Setup GPUs. This will probably by library specific (a la Kokkos).

        # Once setup, add ourselves to the registry
        contexts.append(self)

    def __parla_placement__(self):
        return self.placement

    def __enter__(self):
        r = self.underlying.__enter__()
        assert r == self.underlying
        return self

    def __exit__(self, *args):
        self.underlying.__exit__(*args)

contexts: List[Context] = []

def find_context(placement: List[Device], exact: bool = True):
    placement = frozenset(placement)
    for ctx in contexts:
        if (not exact and placement.issubset(frozenset(ctx.placement))) or \
                (exact and placement == frozenset(ctx.placement)):
            return placement
    return None


class NoopContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


noop_context = NoopContext()