import logging
import os

import pytest

from parla.task_runtime import Scheduler

import parla.cpucores

try:
    import parla.cuda
except:
    pass

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
    with Scheduler(4) as s:
        yield s
