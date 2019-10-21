import logging
from queue import SimpleQueue
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from .device import get_all_devices, get_all_architectures, Device

logger = logging.getLogger(__name__)

__all__ = []

thread_contexts = threading.local()


class PerThreadContext:
    def __init__(self, device: Device, scheduler: "Scheduler"):
        self.device = device
        self.scheduler = scheduler
    def __enter__(self):
        thread_contexts.context = self
    def __exit__(self, exception_type, exception_value, traceback):
        delattr(thread_contexts, "context")


def get_device() -> Device:
    return thread_contexts.context.device


class Scheduler:
    def __init__(self):
        self.mutex = threading.Lock()
        self.active = False
        self.counter_mutex = threading.Lock()

        # Need a lock to ensure atomicity of appending/removing from raised_exceptions.
        # Could do this with atomics in a lower level language, or in any language that is less concurrency impoverished than Python.
        self.exception_log_mutex = threading.Lock()
        self.raised_exceptions = []

    def __enter__(self):
        self.active = True
        self.main_queue = SimpleQueue()
        self.device_queues = {device: SimpleQueue() for device in get_all_devices()}
        self.architecture_queues = {arch: SimpleQueue() for arch in get_all_architectures()}
        self.tasks_in_progress = 0

    def __exit__(self, exception_type, exception_value, traceback):
        self.active = False
        del self.main_queue
        del self.device_queues
        del self.architecture_queues
        del self.tasks_in_progress

    def get(self):
        with self.mutex:
            assert self.active
            device_queue = self.device_queues[get_device()]
            if not device_queue.empty():
                return device_queue.get()
            architecture_queue = self.architecture_queues[get_device().architecture]
            if not architecture_queue.empty():
                return architecture_queue.get()
            if not self.main_queue.empty():
                return self.main_queue.get()
            return None

    def put(self, task, queue_identifier=None):
        """
        Register a task as ready.
        For the time being, queue identifier can be either a thread id,
        in which case the created work item will be pushed onto
        the queue for that thread,
        a device type name, i.e. "cpu" or "gpu"
        in which case the created work item will be pushed onto
        the queue for that device type,
        or None, in which case the work item is pushed onto the main queue.
        """
        with self.mutex:
            assert self.active
            if queue_identifier in get_all_architectures():
                # logger.info("Adding to device queue %r: %r", queue_identifier, self)
                self.architecture_queues[queue_identifier].put(task)
            elif queue_identifier is None:
                # logger.info("Adding to main queue: %r", queue_identifier, self)
                self.main_queue.put(task)
            else:
                # logger.info("Adding to local queue %r: %r", queue_identifier, self)
                self.device_queues[queue_identifier].put(task)

    def run_next(self):
        next_task = self.get()
        if next_task is None:
            return False
        with self.counter_mutex:
            self.tasks_in_progress += 1
        try:
            logger.debug("Running task {} on thread ({})".format(next_task, get_device()))
            next_task.run()
        finally:
            with self.counter_mutex:
                self.tasks_in_progress -= 1
        return True

    def finished(self):
        with self.counter_mutex:
            if self.tasks_in_progress:
                return False
        with self.mutex:
            assert self.active
            for queue in self.device_queues.values():
                if not queue.empty():
                    return False
            for queue in self.architecture_queues.values():
                if not queue.empty():
                    return False
            if not self.main_queue.empty():
                return False
        logger.debug("Exiting %r thread with (%r, %r, %r)", get_device(),
                     map(lambda q: q.qsize(), self.device_queues), self.main_queue.qsize(), self.tasks_in_progress)
        return True


pool_running = False

# TODO: exception handling should just be built into
# (or as a wrapper around) whatever we are using as a thread pool.
def local_func(arg):
    scheduler, thread_id, device = arg
    with PerThreadContext(device, scheduler):
        logger.debug("Starting worker thread: {}".format(arg))
        while not scheduler.finished():
            with scheduler.exception_log_mutex:
                if scheduler.raised_exceptions:
                    logger.debug("Exiting with exceptions: {}".format(scheduler.raised_exceptions))
                    break
            try:
                did_work = scheduler.run_next()
                if not did_work:
                    time.sleep(50 / 1000)
            except Exception as exc:
                with scheduler.exception_log_mutex:
                    scheduler.raised_exceptions.append(exc)
    return

# Note: tasks can be implemented as lock free, however,
# atomics aren't really a thing in Python, so instead
# make each task have its own lock to mimic atomic-like
# counters for dependency tracking.

class Task:
    def __init__(self, func, inputs, dependencies, queue_identifier, scheduler):
        self.func = func
        self.inputs = inputs
        self.remaining_dependencies = len(dependencies)
        self.dependees = []
        self.completed = False
        self.result = None
        self.queue_identifier = queue_identifier
        self.mutex = threading.Lock()
        self.scheduler = scheduler
        with self.mutex:
            for dep in dependencies:
                if dep.completed:
                    self.remaining_dependencies -= 1
                else:
                    dep.dependees.append(self)
            if not self.remaining_dependencies:
                self.enqueue()

    def enqueue(self):
        self.scheduler.put(self, queue_identifier=self.queue_identifier)

    def run(self):
        # logger.debug("Running on %r (should be queue %r): %r", get_device(), self.queue_identifier, self)
        self.func(self, *self.inputs)
        with self.mutex:
            self.completed = True
            for dependee in self.dependees:
                dependee.remaining_dependencies -= 1
                if not dependee.remaining_dependencies:
                    dependee.enqueue()

    def __await__(self):
        return (yield (None, [self], self))

    def __repr__(self):
        return "{func}{inputs}<{remaining_dependencies}, {completed}, queue_id={queue_identifier}>".format(**self.__dict__)

# Lazily starting the thread pool like this still requires the code
# to be organized so that there's a single "generation" task
# even though separate functions aren't necessary anymore.
# Maybe launching/stopping the thread pool would be better as a
# context manager.

# AMP: I think we just need to chuck the threadpool, so that we
#  can move to a customized threading system which can lazily
#  initialize and then prevent interpreter exit by having
#  non-daemon threads. The pool can shutdown based on inactivity
#  if we want, but it will use very few resources because we can
#  have "new work" condition that the threads block on and run_task
#  asserts if it is called from a non-pool thread.

def run_task(func, inputs, dependencies, queue_identifier = None):
    global pool_running
    if pool_running:
        return Task(func, inputs, dependencies, queue_identifier, thread_contexts.context.scheduler)
    else:
        # TODO: Do we want an interface that lets the scheduler be specified at
        # runtime instead of just having it be here?
        scheduler = Scheduler()
        devices = get_all_devices()
        pool_running = True
        with scheduler, ThreadPoolExecutor(len(devices)) as pool:
            root_task = Task(func, inputs, dependencies, queue_identifier, scheduler)
            local_loops = pool.map(local_func, map(lambda x: (scheduler, x[0], x[1]), enumerate(devices)))
            assert all(map(lambda x: x is None, local_loops))
        pool_running = False
        with scheduler.exception_log_mutex:
            if scheduler.raised_exceptions:
                # TODO: Handle multiple exception case better
                exc = scheduler.raised_exceptions[0]
                raised_exceptions = []
                raise Exception("An error occurred in a worker thread.") from exc
        pool_running = False
        return root_task
