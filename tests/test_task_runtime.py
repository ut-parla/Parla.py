import parla.cpu
from parla.task_runtime import run_task, Task

def test_flag_increment():
    external_flag = 0
    def increment_flag(task):
        nonlocal external_flag
        external_flag += 1
    tsk = run_task(increment_flag, tuple(), [])
    assert external_flag

def test_deps():
    def tasks_with_deps(task):
        counter = 0
        def increment(task):
            nonlocal counter
            counter += 1
        def check(task):
            assert counter == 1
        first = run_task(increment, tuple(), [])
        second = run_task(check, tuple(), [])
    run_task(tasks_with_deps, tuple(), [])

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
                run_task(recurse, (val-1,), [])
                run_task(recurse, (val-1,), [])
        run_task(recurse, [2], [])
    run_task(recursion_without_continuation, tuple(), [])

def test_recursion_with_finalization():
    outermost = []
    counter = 0
    def recursion_with_manual_continuation(task, val):
        if val == 0:
            nonlocal counter
            counter += 1
        else:
            t1 = run_task(recursion_with_manual_continuation, (val-1,), [])
            t2 = run_task(recursion_with_manual_continuation, (val-1,), [])
            if val == 1:
                nonlocal outermost
                outermost.append(t1)
                outermost.append(t2)
    run_task(recursion_with_manual_continuation, (3,), [])
    def check_counter(task):
        assert counter == 8
    run_task(check_counter, tuple(), outermost)

def test_exception_handling():
    class CustomException(Exception):
        pass
    def raise_exc(task):
        raise CustomException("error")
    try:
        tsk = run_task(raise_exc, tuple(), [])
    except:
        success = True
    else:
        success = False
    assert success

def test_exception_handling_state_restoration():
    test_flag_increment()
    test_exception_handling()
    test_flag_increment()

