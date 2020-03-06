from parla import multiload, multiload_context

def test_multiload():
    with multiload():
        import multiload_test_module as mod
    mod.unused_id = None
    assert not mod._parla_load_id
    with multiload_context(1):
        assert 1 == mod._parla_load_id
        assert getattr(mod, "_parla_forwarding_module", False)
        # Need forbiddenfruit to make this last one work.
        #assert not hasattr(mod, "unused_id")
