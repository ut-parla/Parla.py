from . import mutual_2

def check():
    assert getattr(mutual_2, "_parla_forwarding_module", False)
