import subprocess


def call_example(name):
    proc = subprocess.run(["python", "-m", "examples." + name], check=True)
    print(proc)


def test_fox():
    call_example("fox")


def test_inner():
    call_example("inner")


def test_jacobi():
    call_example("jacobi")


def test_blocked_cholesky():
    call_example("blocked_cholesky")


def test_conjugate_gradient():
    call_example("conjugate_gradient")
