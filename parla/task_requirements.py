# General imports
import logging
from abc import abstractmethod, ABCMeta
from itertools import combinations
from typing import Collection, Union, Dict, List, Any, FrozenSet, Iterable

# Parla imports
from parla.device import Device
from parla.environments import TaskEnvironment

# Logger configuration (uncomment and adjust level if needed)
#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

__all__ = ["DeviceSetRequirements", "OptionsRequirements", "ResourceRequirements"]

ResourceDict = Dict[str, Union[float, int]]
class ResourceRequirements(object, metaclass=ABCMeta):
    """
    When a task spawns, it has a set of requirements based on parameters
    supplied to @spawn.
    This class represents those resources.
    This is an Abstract Base Class - see below for classes which inherit from it.
    Currently, spawned tasks only use DeviceSetRequirements.
    After mapping, tasks receive EnvironmentRequirements.
    As of writing this comment, idk what the difference is. Enviroments seem unnecessary and confusing.
    OptionsRequirements aren't even used anywhere at all.
    """
    __slots__ = ["resources", "ndevices", "tags"]

    tags: FrozenSet[Any]
    resources: ResourceDict
    ndevices: int
    devices: FrozenSet[Device]

    def __init__(self, resources: ResourceDict, ndevices: int, tags: Collection[Any]):
        assert all(isinstance(v, str) for v in resources.keys())
        assert all(isinstance(v, (float, int)) for v in resources.values())
        self.resources = resources
        self.ndevices = ndevices
        self.tags = frozenset(tags)

    @property
    def possibilities(self) -> Iterable["ResourceRequirements"]:
        return [self]

    @property
    def exact(self):
        return False

    @abstractmethod
    def __parla_placement__(self):
        raise NotImplementedError()


class EnvironmentRequirements(ResourceRequirements):
    __slots__ = ["environment"]
    environment: TaskEnvironment

    def __init__(self, resources: ResourceDict, environment: TaskEnvironment, tags: Collection[Any]):
        super().__init__(resources, len(environment.placement), tags)
        self.environment = environment

    @property
    def devices(self):
        return self.environment.placement

    @property
    def exact(self):
        return True

    def __parla_placement__(self):
        return self.environment.__parla_placement__()

    def __repr__(self):
        return "EnvironmentRequirements({}, {})".format(self.resources, self.environment)


# This basically stores all the devices a task is *permitted* to run on,
# taking into account spawn's placement parameter
class DeviceSetRequirements(ResourceRequirements):
    __slots__ = ["devices"]
    devices: FrozenSet[Device]

    def __init__(self, resources: ResourceDict, ndevices: int, devices: Collection[Device], tags: Collection[Any]):
        super().__init__(resources, ndevices, tags)
        assert devices
        assert all(isinstance(dd, Device) for dd in devices)
        self.devices = frozenset(devices)
        assert len(self.devices) >= self.ndevices

    @property
    def possibilities(self) -> Iterable["DeviceSetRequirements"]:
        return (DeviceSetRequirements(self.resources, self.ndevices, ds, self.tags)
                for ds in combinations(self.devices, self.ndevices))

    @property
    def exact(self):
        return len(self.devices) == self.ndevices

    def __parla_placement__(self):
        return self.devices

    def __repr__(self):
        return "DeviceSetRequirements({}, {}, {}, exact={})".format(self.resources, self.ndevices, self.devices, self.exact)


# CURRENTLY NOT USED
class OptionsRequirements(ResourceRequirements):
    __slots__ = ["options"]
    options: List[List[Device]]

    def __init__(self, resources, ndevices, options, tags: Collection[Any]):
        super().__init__(resources, ndevices, tags)
        assert len(options) > 1
        assert all(isinstance(a, Device) for a in options)
        self.options = options

    @property
    def possibilities(self) -> Iterable[DeviceSetRequirements]:
        return (opt
                for ds in self.options
                for opt in DeviceSetRequirements(self.resources, self.ndevices, ds, self.tags).possibilities)

    def __parla_placement__(self):
        return list(set(d for ds in self.options for d in ds))

    def __repr__(self):
        return "OptionsRequirements({}, {}, {})".format(self.resources, self.ndevices, self.options)


