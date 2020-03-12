//
// Created by amp on 3/6/20.
//

#ifndef NESTED_RUNTIMES_VIRT_DLOPEN_H
#define NESTED_RUNTIMES_VIRT_DLOPEN_H

typedef struct {
    /// Iff true, override dlopen with dlmopen on the `lm` specified below.
    char enabled;
    /// The LM to use with dlmopen. This is ignored if `!enabled`.
    long int lm;
} virt_dlopen_state;

/// A variable initializer for states which starts them disabled.
#define VIRT_DLOPEN_STATE_INITIALIZER {0, LM_ID_BASE}

/// Get the state of the virtual dlopen library. This state is thread-local.
virt_dlopen_state virt_dlopen_get_state();

/// Set the state of the virtual dlopen library and return the old state. The returned value can be restored later to
/// get stack semantics. This state is thread-local.
virt_dlopen_state virt_dlopen_swap_state(char enabled, long int lm);

#endif //NESTED_RUNTIMES_VIRT_DLOPEN_H
