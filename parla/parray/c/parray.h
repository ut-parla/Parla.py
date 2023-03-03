#pragma once

#include <unordered_map>
#include "parray_state.h"

namespace parray {
    // PArray C++ interface which provides some information that will be used for scheduling task
    class PArray{
        public:
            uint64_t id;  // unique ID of the PArray

            PArray();
            PArray(uint64_t, PArrayState *);

            // Get current size (in bytes) of each copy of the PArray
            // if it is a subarray, return the subarray's size
            uint64_t get_size();

            // Set the size of the PArray
            void set_size(uint64_t new_size);

            // Return True if there is an PArray copy (possibly invalid) on this device
            bool exists_on_device(uint64_t device_id);

            // Return True if there is an PArray copy and its coherence state is valid on this device
            bool valid_on_device(uint64_t device_id);

        private:
            uint64_t _size;  // number of bytes consumed by each copy of the array/subarray
            PArrayState* _state;  // state of a PArray (subarray share this object with its parent)
    };
}