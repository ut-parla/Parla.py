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

def test_multiple_contexts():
    import timeit
    multiload_context(1).set_allowed_cpus([0])
    multiload_context(2).set_allowed_cpus([1,2,3,4,5,6,7])
    with multiload():
        import numpy as np
    def timed_thing():
        np.int(1)
        return 0
        # return timeit.timeit(lambda: np.dot(np.random.rand(2000, 2000), np.random.rand(2000, 2000)), number=1)
    with multiload_context(1):
        ctx_1 = timed_thing()
    with multiload_context(2):
        ctx_2 = timed_thing()
    # assert ctx_2 > ctx_1*1.5
