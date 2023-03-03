#include <unordered_map>
#include "parray_state.h"

namespace parray {
    PArrayState::PArrayState() {}

    bool PArrayState::exists_on_device(int device_id) {
        if (auto findit = this->_exist_on_device.find(device_id); findit != this->_exist_on_device.end()) {
            return findit->second;
        } else {
            return false;
        }
    }

    bool PArrayState::valid_on_device(int device_id) {
        if (auto findit = this->_valid_on_device.find(device_id); findit != this->_valid_on_device.end()) {
            return findit->second;
        } else {
            return false;
        }
    }

    void PArrayState::set_exist_on_device(int device_id, bool exist) {
        this->_exist_on_device[device_id] = exist;
    }

    void PArrayState::set_valid_on_device(int device_id, bool valid) {
        this->_valid_on_device[device_id] = valid;
    }
}