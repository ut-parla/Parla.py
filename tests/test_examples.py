import os
import subprocess

from pytest import skip


def call_example(name):
    env = dict(os.environ)
    env["TESTING"] = "true"
    proc = subprocess.run(["python", "-m", "examples." + name], check=True, env=env)


def test_fox():
    call_example("fox")


def test_inner():
    call_example("inner")


def test_jacobi():
    call_example("jacobi")


def test_blocked_cholesky():
    skip("Test too slow.")
    call_example("blocked_cholesky")


def test_conjugate_gradient():
    call_example("conjugate_gradient")
