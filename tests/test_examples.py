import os
import subprocess

from pytest import skip


def call_example(name):
    env = dict(os.environ)
    env["TESTING"] = "true"
    proc = subprocess.run(["python", "-m", "examples." + name], check=True, env=env)


def test_fox():
    try:
        from parla.cuda import gpu
    except:
        skip("Test needs cuda.")
        return
    call_example("fox")


def test_inner():
    try:
        from parla.cuda import gpu
    except:
        skip("Test needs cuda.")
        return
    call_example("inner")


def test_jacobi():
    try:
        from parla.cuda import gpu
    except:
        skip("Test needs cuda.")
        return
    call_example("jacobi")


def test_blocked_cholesky():
    try:
        from parla.cuda import gpu
    except:
        skip("Test needs cuda.")
        return
    call_example("blocked_cholesky")
