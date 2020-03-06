from multiload_test_module.absolute import a
from .relative import b, c, d

assert a is d
assert b == "b"
assert c == "c"
