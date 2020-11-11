from multiload_test_module.absolute import a
from .relative import b, c, d
from . import mutual_1, mutual_2

assert a is d
assert b == "b"
assert c == "c"
