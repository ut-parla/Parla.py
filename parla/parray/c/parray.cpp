#include <unordered_map>
#include <cstdint>
#include "parray.h"

namespace parray {
    PArray::PArray() : id(-1), _state(nullptr) {}

    PArray::PArray(uint64_t id, PArrayState* state) : id(id), _state(state) {}

    uint64_t PArray::get_size() {
        return this->_size;
    }

    void PArray::set_size(uint64_t new_size) {
        this->_size = new_size;
    }

    bool PArray::exists_on_device(uint64_t device_id) {
        return this->_state->exists_on_device(device_id);
    }

    bool PArray::valid_on_device(uint64_t device_id) {
        return this->_state->valid_on_device(device_id);
    }
}