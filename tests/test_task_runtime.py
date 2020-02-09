from parla import get_all_devices
from parla.cpu import cpu
from parla.task_runtime import Scheduler, Task, TaskCompleted, TaskRunning, DeviceSetRequirements
from parla.tasks import TaskID

task_id_next = 0


def simple_task(func, args=(), dependencies=(), taskid=None, devices=list(get_all_devices())):
    global task_id_next
    taskid = taskid or TaskID("Dummy {}".format(task_id_next), task_id_next)
    task_id_next += 1
    return Task(func, args, dependencies, taskid, req=DeviceSetRequirements(devices=devices, ndevices=1, resources={}))


def test_flag_increment():
    external_flag = 0
    def increment_flag(task):
        nonlocal external_flag
        external_flag += 1
    with Scheduler(4):
        simple_task(increment_flag, tuple(), [])
    assert external_flag


def test_deps():
    external_flag = 0
    with Scheduler(4):
        def tasks_with_deps(task):
            counter = 0
            def increment(task):
                nonlocal counter
                counter += 1
            first = simple_task(increment)
            def check(task):
                assert counter == 1
                nonlocal external_flag
                external_flag += 1
            simple_task(check, dependencies=[first])
        simple_task(tasks_with_deps)
    assert external_flag

def test_recursion_without_continuation():
    def recursion_without_continuation(task):
        counter = 0
        counter_vals = [True, True, True, True]
        def recurse(task, val):
            if val == 0:
                nonlocal counter
                assert counter_vals[counter]
                counter_vals[counter] = False
                counter += 1
            else:
                simple_task(recurse, (val-1,), [])
                simple_task(recurse, (val-1,), [])
        simple_task(recurse, [2], [])
    with Scheduler(4):
        simple_task(recursion_without_continuation, tuple(), [])

def test_recursion_with_finalization():
    counter = 0
    with Scheduler(4):
        def recursion_with_manual_continuation(task, val):
            if val == 0:
                nonlocal counter
                counter += 1
                return TaskCompleted(None)
            else:
                t1 = simple_task(recursion_with_manual_continuation, (val-1,), [])
                t2 = simple_task(recursion_with_manual_continuation, (val-1,), [])
                def k(task):
                    if val == 3:
                        assert counter == 8
                    return TaskCompleted(None)
                return TaskRunning(k, (), [t1, t2])
        simple_task(recursion_with_manual_continuation, (3,), [])


def test_exception_handling():
    class CustomException(Exception):
        pass
    def raise_exc(task):
        raise CustomException("error")
    try:
        with Scheduler(4):
            simple_task(raise_exc, tuple(), [])
    except:
        success = True
    else:
        success = False
    assert success


def test_exception_handling_state_restoration():
    test_flag_increment()
    test_exception_handling()
    test_flag_increment()


# TODO: Add tests for ResourcePool. Sadly they will need to be multithreaded tests since correct blocking
#  is very important. :-/ PITA.