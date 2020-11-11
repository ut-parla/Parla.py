from . import mutual_1

def check():
    assert getattr(mutual_1, "_parla_forwarding_module", False)
