"""
Parla `TaskEnvironments` represent execution environments that a task can run in.
Each environment is assigned resources that tasks in that environment will use.
"""

from abc import ABCMeta, abstractmethod
from typing import Collection, Union, Any, List, ContextManager, Dict, Callable, Iterator, FrozenSet, Tuple, \
    Iterable

from .device import Architecture, Device

__all__ = ["TaskEnvironment"]

class EnvironmentComponentInstance(ContextManager, metaclass=ABCMeta):
    """
    A component of a TaskEnvironment which provides some services to tasks.

    EnvironmentComponents are constructed using EnvironmentComponentDescriptors to allow the configuration to be
    manipulated explicitly before construction.

    Once part of a TaskEnvironment, EnvironmentComponents help configure the task execution environment and can be
    directly accessed by tasks. Access is done via the descriptor type.

    The component itself is a context manager which will configure the thread execution environment to use this
    component.
    """

    """
    The descriptor used to create this component.
    """
    descriptor: "EnvironmentComponentDescriptor"

    def __init__(self, descriptor: "EnvironmentComponentDescriptor"):
        self.descriptor = descriptor

    @abstractmethod
    def initialize_thread(self) -> None:
        """
        Initialize the current thread for this component.
        """
        raise NotImplementedError()

class EnvironmentComponentDescriptor(Callable[[], EnvironmentComponentInstance], metaclass=ABCMeta):
    """
    A descriptor for an EnvironmentComponent.

    The descriptor can be combined with other descriptors of the same type and
    can be used to construct actual components.
    """

    @abstractmethod
    def __call__(self, env: "TaskEnvironment") -> EnvironmentComponentInstance:
        """
        Construct a concrete EnvironmentComponent based on this descriptor.
        """
        raise NotImplementedError()

    @abstractmethod
    def combine(self, other):
        """
        Combine two descriptors of the same type.
        :param other: Another EnvironmentComponentDescriptor with the same type as self.
        :return: a new EnvironmentComponentDescriptor which combines self and other.
        """
        raise NotImplementedError()

class TaskEnvironment(ContextManager):
    tags: FrozenSet[Any]
    placement: FrozenSet[Device]
    components: Dict[EnvironmentComponentDescriptor, EnvironmentComponentInstance]

    def __init__(self,
                 placement: Collection[Union[Architecture, Device, "parla.tasks.Task", "parla.tasks.TaskID", Any]],
                 components: Collection[EnvironmentComponentDescriptor] = None,
                 tags: Collection[Any] = ()):
        """
        Create a new task execution environment which will run with the given placement.
        :param placement: A placement list containing devices and architectures.
        :param components: The components that should be used for this environment,
            or None meaning the default components should be used.
        :param tags: A set of arbitrary tags associated with this environment. Tasks can select environments by tag.
        """
        from .tasks import get_placement_for_set

        tags = frozenset(tags)
        try:
            hash(tags)
        except TypeError as e:
            raise TypeError("TaskEnvironment tags must be hashable", e)
        self.tags = tags

        self.placement = get_placement_for_set(placement)

        if components is None:
            components = [c for d in self.placement for c in d.default_components]
        components = TaskEnvironment._combine_like_components(components)
        self.components = {type(c): c(self) for c in components}

    def __parla_placement__(self):
        return self.placement

    def __enter__(self):
        for c in self.components.values():
            c.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for c in self.components.values():
            c.__exit__(exc_type, exc_val, exc_tb)

    def __repr__(self):
        return "TaskEnvironment({}, {}, {})".format(self.placement, self.components, self.tags)

    def __hash__(self):
        return id(self)

    @classmethod
    def _combine_like_components(cls, components):
        """
        :param components: An iterable of EnvironmentComponentDescriptors.
        :return: A list of EnvironmentComponentDescriptors in which each type only appears once and the instance is a
            combination of all isntances in `components`.
        """
        out = {}
        for c in components:
            if type(c) in out:
                out[type(c)] = out[type(c)].combine(c)
            else:
                out[type(c)] = c
        return list(out.values())

    def get_events_from_components(self) -> List:
        # This function aggregates all events created by each component,
        # and returns back to the task_runtime. Through this, dependees
        # can wait for them on the proper devices.
        # Note that this function returns a list of a pair of (Arch type string,
        # event object).
        events = []
        for c in self.components.values():
            event = c.get_event_object()
            if event is not None:
                events.append(event)
        return events

    def wait_dependent_events(self, events: List):
        for event_info in events:
            # TODO(lhc): should be refactored
            dev_type, event = event_info[0]
            for c in self.components.values():
                if c.check_device_type(dev_type):
                    c.wait_event(event)
                    break

    def record_events(self):
        for c in self.components.values():
            c.record_event()

    def sync_events(self):
        for c in self.components.values():
            c.sync_event()


class TaskEnvironmentRegistry(Collection[TaskEnvironment]):
    """
    A collections of task environments with a utility to look up environments based on their placement and tags.
    """

    task_environments: List[TaskEnvironment]

    def __init__(self, *envs):
        self.task_environments = list(envs)

    def __iter__(self) -> Iterator[TaskEnvironment]:
        return iter(self.task_environments)

    def __contains__(self, e) -> bool:
        return e in self.task_environments

    def __len__(self) -> int:
        return len(self.task_environments)

    # def register(self, *envs: "TaskEnvironment"):
    #     env = None
    #     for e in envs:
    #         assert isinstance(e, TaskEnvironment)
    #         self.task_environments.append(e)
    #         env = e
    #     return env

    # def find(self, placement: Collection[Device], tags: Collection[Any], exact: bool) -> TaskEnvironment:
    #     return next(iter(self.find_all_ordered(placement, tags, exact)), None)

    def _find_all(self, placement: Collection[Device], tags: Collection[Any], exact: bool) -> \
            Iterable[Tuple[TaskEnvironment, int]]:
        placement = frozenset(placement)
        tags = frozenset(tags)
        for env in self.task_environments:
            env_placement = frozenset(env.placement)
            env_tags = frozenset(env.tags)
            if (not exact and placement.issubset(env_placement) and tags.issubset(env_tags)) or \
                    (exact and placement == env_placement and tags == env_tags):
                yield (env, len(env_placement - placement))

    def find_all(self, placement: Collection[Device], tags: Collection[Any], exact: bool) -> \
            Iterable[TaskEnvironment]:
        return (e for e, _ in self._find_all(placement, tags, exact))

    def find_all_ordered(self, placement: Collection[Device], tags: Collection[Any], exact: bool) -> \
            List[TaskEnvironment]:
        l = list(self._find_all(placement, tags, exact))
        l.sort(key=lambda t: t[1])
        return [e for e, _ in l]
