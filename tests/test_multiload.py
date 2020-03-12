from parla.multiload import multiload, multiload_context

def test_multiload():
    with multiload():
        import multiload_test_module as mod
    mod.unused_id = None
    assert not mod._parla_load_context.nsid
    with multiload_context(1):
        assert multiload_context(1) == mod._parla_load_context
        assert getattr(mod, "_parla_forwarding_module", False)
        # Need forbiddenfruit to make this last one work.
        #assert not hasattr(mod, "unused_id")
