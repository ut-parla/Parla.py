import logging
import os

import pytest

from parla import TaskEnvironment
from parla.task_runtime import Scheduler

if os.getenv("LOG_LEVEL") is not None:
    level = os.getenv("LOG_LEVEL")
    logging.basicConfig(level=level)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    parla.tasks.logger.setLevel(level)
    parla.task_runtime.logger.setLevel(level)
else:
    logging.basicConfig(level=logging.INFO)

@pytest.fixture
def runtime_sched():
    from parla.cpu import cpu

    # Dummy environments with no components for testing.
    environments = [TaskEnvironment(placement=[d], components=[]) for d in cpu.devices]

    with Scheduler(environments, 4) as s:
        yield s
