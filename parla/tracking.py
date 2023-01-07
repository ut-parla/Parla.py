"""
Contains data structures for logging and tracking PArray instances across devices.

DataTracker:
- Hash-map of PArrays to devices
- Hash-map of Devices to EvictionManagers

EvictionManager:
 - Implements an eviction policy for a managed list of PArray objects on a single device 
"""

#TODO(hc): how to guarantee that the evicted data can be refetched correctly when a later task needs it.
#          synchronization between data movement tasks and data eviction; for example, if a data is evicted
#          after a data movement task is executed, then it breaks correctness.
#          also, it could cause thrashing.
#          condition 1. if a mapped task in a queue will use an evicted data, its data movement task should exist.
#          condition 2. 

#TODO(hc): so, eviction should happen at the proper timing.
#          let's assume that we evicted a proper data at a proper timing. then what do we need?
#          1) parray coherency protocol should be aware of that.
#          2) update priority of the data object; or we can just remove that data from the list.

#TODO(hc): how to handle slicing?


import threading


#TODO(wlr): Nothing in this file is threadsafe at the moment. Developing structure first, then we'll add locks.

#TODO(wlr): I assume PArrays hash to a unique value during their lifetime. If not, we'll need to add such a hash function to PArray.

#TODO(wlr): This is developed without considering sliced PArrays.
# For slices, I imagine we might need something with the following rules:
#   - An access of a slice, locks the parent on that device
#   - An eviction of a slice, may not evict the parent
#   - An eviction of a parent, evicts all slices

# I'm less certain about the following:
#   - Updating the priority of a slice, updates the priority of the parent
#   - Updating the priority of a parent, updates the priority of all slices


#Needs:
# - wrap in locks (use external locks on base class)

from enum import Enum
from typing import TypedDict, Dict
from parla.cpu_impl import cpu


# TODO(hc): It should be declared on PArray.
class DataNode:
    """
    A node containing a data object (PArray) on a specific device. 
    Used in the linked lists in the EvictionManager
    """
    def __init__(self, data, device, priority=0):
        self._data = data
        self._device = device
        self._priority = priority

        self.next = None
        self.prev = None

    @property
    def data(self):
        return self._data

    @property
    def device(self):
        return self._device

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"DataNode({self.data}, {self.device})"


class ListNode:
    """
    A node containing a linked list of DataNodes with an associated value (typically priority).
    Useful for more complex eviction data structures.
    """

    def __init__(self, value, list):
        self.value = value
        self.list = list

        self.next = None
        self.prev = None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"ListNode({self.list})"


# TODO(hc): This list should be protected by a lock.
class DLList:
    """
    A doubly linked list used in the EvictionManager.
    """
    def __init__(self):
        self.head = None
        self.tail = None
        self.next = None
        self.prev = None
        self.length = 0
        self._list_lock = threading.Condition(threading.Lock())

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"DLList({self.head}, {self.tail})"

    def append(self, node):
        with self._list_lock:
            if self.head is None:
                self.head = node
                self.tail = node
            else:
                self.tail.next = node
                node.prev = self.tail
                self.tail = node
            self.length += 1

    def remove(self, node):
        with self._list_lock:
            edit = False

            if self.head == node:
                self.head = node.next
                edit = True

            if self.tail == node:
                self.tail = node.prev
                edit = True

            if node.prev is not None:
                node.prev.next = node.next
                edit = True

            if node.next is not None:
                node.next.prev = node.prev
                edit = True

            node.prev = None
            node.next = None

            if edit:
                self.length -= 1

            return edit
        
    def insert_before(self, node, new_node):
        with self._list_lock:
            if node.prev is not None:
                node.prev.next = new_node
                new_node.prev = node.prev
            else:
                self.head = new_node
            node.prev = new_node
            new_node.next = node

            self.length += 1

    def insert_after(self, node, new_node):
        with self._list_lock:
            if node.next is not None:
                node.next.prev = new_node
                new_node.next = node.next
            else:
                self.tail = new_node
            node.next = new_node
            new_node.prev = node

            self.length += 1

    def __len__(self):
        with self._list_lock:
            return self.length

    def __repr__(self):
        with self._list_lock:
            repr_str = "<Data List>:\n"
            tmp_node = self.head
            while (tmp_node != None):
                repr_str += str(id(tmp_node)) + " -> "
                tmp_node = tmp_node.next
            repr_str += "\n"
            return repr_str


class GCDataState(Enum):
    """
    Enum of data states.
    """
    PREFETCHING = "Prefetching" # Data is being prefetched.
    RESERVED = "Reserved" # Data's ref. count is >= 1, but not acquired.
    ACQUIRED = "Acquired" # A task is using data.
    FREE = "Free" # None of tasks (mapped/running) does not need data.


class DataMapType(TypedDict):
    """
    Track information of data instance in a device
    """
    # TODO(hc): state should be an enum type.
    state: GCDataState
    ref_count: int
    ref_list_node: DataNode


#class LRUManager(EvictionManager):
class LRUManager:
    """
    LRU policy for garbage collecting.
    It mantains a list of the zero-referenced data objects for each device.
    The head of the list is the target task to be evicted
    and the tail of the list is the data used most recently.
    """

    def __init__(self, memory_limit = 999999):
#super().__init__(device, memory_limit)
        # A list containig zero-reference data objects in a specified device.
        self.zr_data_list = DLList()
        # A dictionary containing all data information on a device.
        self.data_dict: Dict[str, DataMapType] = {}
        # A lock for guarding a data state.
        self._data_state_lock = threading.Condition(threading.Lock())
        # A lock for guarding a reference count.
        self._data_ref_count_lock = threading.Condition(threading.Lock())

        #Note(wlr): These tracking dictionaries are optional, I just think it's interesting to track.
        #Holds data objects on this device that are being prefetched.
        #XXX(hc): This might be necessary as data being prefetched cannot be used yet but it can avoid
        # unnecessary future data prefetching or move.
        self.prefetch_map = {}
        #Holds data objects on this device that are needed by tasks that have not yet completed (this includes data in the process of being prefetched).
        self.active_map = {}
        #holds data objects that are currently being used by tasks.
        self.used_map = {}

    def _increase_ref_count(self, data_info):
        with self._data_ref_count_lock:
            assert(data_info["ref_count"] >= 0)
            data_info["ref_count"] += 1

    def _decrease_ref_count(self, data_info):
        with self._data_ref_count_lock:
            data_info["ref_count"] -= 1
            assert(data_info["ref_count"] >= 0)

    def _check_ref_count_zero(self, data_info):
        with self._data_ref_count_lock:
            print("Check:", data_info["ref_count"], flush=True)
            return data_info["ref_count"] == 0

    def _update_data_state(self, data_id, new_state, taskid):
        with self._data_state_lock:
            # prefetching, reserved, using, free
            data_info = self.data_dict[data_id]
            data_state = data_info["state"]
            print(f"[GC] (Task: {taskid}) Data (ID: {data_id})'s state is updated from "+
                  f"{data_state} to {new_state}", flush=True)
            if data_state == new_state:
                return
            if new_state == GCDataState.PREFETCHING:
                if data_state == GCDataState.FREE:
                    data_info["state"] = new_state
                return
            elif new_state == GCDataState.RESERVED:
                if data_state == GCDataState.PREFETCHING or \
                   data_state == GCDataState.FREE:
                    data_info["state"] = new_state
                return
            elif new_state == GCDataState.ACQUIRED:
                print(">>>>> ", data_state, flush=True)
                assert(data_state == GCDataState.RESERVED)
                data_info["state"] = new_state
                return
            elif new_state == GCDataState.FREE:
                assert(data_state == GCDataState.ACQUIRED)
                data_info["state"] = new_state
                return

    def _dict_id(self, data, dev):
        """ Genereate an ID of a data on a data information dictionary. """
        dev_index = "G" + str(dev.index) if (dev.architecture is not cpu) else "C" 
        return str(data.ID) + "." + dev_index

    def _start_prefetch_data(self, data, dev, taskid = ""):
        data_id = self._dict_id(data, dev)
        if data_id in self.data_dict:
            #This is a prefetch of a data object that is already on the device (or is being prefetched).
            #This means the data is no longer evictable as its about to be in-use by data movement and compute tasks.
            #Remove it from the evictable list.

            # TODO(hc): but if a data movement task will be executed after a very long time, that also can be evictable.
            #           if memory is full and any task cannot proceed, we can still evict one of data that was prefetched.
            #           but this is very rare case and I am gonna leave it as the future work.
            
            # TODO(hc): PArray should point to a corresponding data node.
            data_info = self.data_dict[data_id]
            success = self.zr_data_list.remove(data_info["ref_list_node"])
            self._update_data_state(data_id, GCDataState.PREFETCHING, taskid)

            #if success:
                #This is the first prefetch of a data object that is already on the device.
                #Update the evictable memory size (as this data object is no longer evictable).
                #self.evictable_memory -= data.size
            self._increase_ref_count(data_info)
            print(f"[GC] (Task: {taskid}) Existing data (ID: {data_id}) is updated "+
                f"through prefetching (Ref. count: {self.data_dict[data_id]['ref_count']}, "+
                f"Ref. node ID: {id(self.data_dict[data_id]['ref_list_node'])})", flush=True)
        else:
            self.data_dict[data_id] = { "state" : GCDataState.PREFETCHING, \
                                        "ref_count" : 1, \
                                        "ref_list_node" : DataNode(data, dev) }
            print(f"[GC] (Task: {taskid}) New data (ID: {data_id}) is added through "+
                  f"prefetching (Ref. count: {self.data_dict[data_id]['ref_count']}, "+
                  f"Ref. node ID: {id(self.data_dict[data_id]['ref_list_node'])})", flush=True)
        print(f"[GC] (Task: {taskid}) Zero-referenced list after prefetching data: "+
              f"\n{self.zr_data_list}", flush=True)
            #This is a new block, update the used memory size.
            #self.used_memory += data.size
        #self.prefetch_map[data] = data
        #self.active_map[data] = data
        #assert(self.used_memory <= self.memory_limit)

    def _stop_prefetch_data(self, data, dev, taskid=""):
        data_id = self._dict_id(data, dev)
        assert(data_id in self.data_dict)
        self._update_data_state(data_id, GCDataState.RESERVED, taskid) 
        print(f"[GC] (Task: {taskid}) Existing data (ID: {data_id}) is updated as "+
            f"prefetching completes (Ref. count: {self.data_dict[data_id]['ref_count']}, "+
            f"Ref. node ID: {id(self.data_dict[data_id]['ref_list_node'])})", flush=True)


    def _acquire_data(self, data, dev, taskid = ""):
        #NOTE(wlr): The data should already be removed from the evictable list in the prefetching stage.
        #           Any logic here would be a sanity check. I'm removing it for now.
        #node = self.data_dict.get(data, None)
        #if node is not None:
        #    #Remove the node from the eviction list while it's in use
        #    success = self.data_list.remove(node)
        #self.used_map[data] = data
        #data.add_use(self.device)
        data_id = self._dict_id(data, dev)
        assert(data_id in self.data_dict)
        self._update_data_state(data_id, GCDataState.ACQUIRED, taskid) 
        print(f"[GC] (Task: {taskid}) Existing data (ID: {data_id}) is updated as "+
            f"a compute task acquires"+
            f" (Ref. count: {self.data_dict[data_id]['ref_count']}, "+
            f"Ref. node ID: {id(self.data_dict[data_id]['ref_list_node'])})", flush=True)


    def _release_data(self, data, dev, taskid = ""):
        data_id = self._dict_id(data, dev)
        assert(data_id in self.data_dict)
        data_info = self.data_dict[data_id]
        self._decrease_ref_count(data_info)

        #active_count = data.get_active(self.device)
        #use_count = data.get_use(self.device)

        #data.remove_active(self.device)
        #data.remove_use(self.device)

        if data_info["ref_count"] == 0:
            assert(self._check_ref_count_zero(data_info))
            #del self.active_map[data]
            #If the data object is no longer needed by any already prefetched tasks, it can be evicted.
            node = data_info["ref_list_node"]
            self.zr_data_list.append(node)
            #self.evictable_memory += data.nbytes
        #if use_count == 1:
            #del self.used_map[data]
        print(f"[GC] (Task: {taskid}) Existing data (ID: {data_id}) is updated as a compute "+
            f"task releases (Ref. count: {self.data_dict[data_id]['ref_count']}, "+
            f"Ref. node ID: {id(self.data_dict[data_id]['ref_list_node'])})", flush=True)
        print(f"[GC] (Task: {taskid}) Zero-referenced list after releasing data: "+
              f"\n{self.zr_data_list}", flush=True)

    def _evict_data(self, target_data, target_dev):
        data_id = self._dict_id(target_data, target_dev)
        data_info = self.data_dict[data_id]
        assert(self._check_ref_count_zero(data_info))
        #Call internal data object evict method
        #This should:
        #  - Backup the data if its not in a SHARED state
        #    (SHARED state means the data has a valid copy on multiple devices. Eviction should never destroy the only remaining copy)
        #  - Mark the data for deletion (this may be done by the CuPy/Python GC)
        target_data.evict(target_dev)
        self.zr_data_list.remove(data_info["ref_list_node"])
        del data_info
        #self.used_memory -= data.nbytes
        #self.evictable_memory -= data.nbytes

    def _evict(self):
        # Get the oldest data object
        # Because we append after use this is at the front of the list
        node = self.zr_data_list.head
        n_data = node.data
        n_dev = node.dev 
        self._evict_data(n_data, n_dev)

'''
class LFUManager(EvictionManager):
    """
    Eviction Manager for a LFU (Least Frequently Used) policy. 
    Use is updated when a data object is accessed and released by a task.

    The data structure follows the O(1) implementation described by (Ketan Shah/Anirban Mitra/Dhruv Matani, 2010):  http://dhruvbird.com/lfu.pdf
    """

    def __init__(self, device, memory_limit):
        super().__init__(device, memory_limit)
        self.priority_map = {}
        self.data_map = {}

        self.priority_list = DLList()

        #NOTE(wlr): These tracking dictionaries are optional, I just think it's interesting to track.
        #Holds data objects on this device that are being prefetched.
        self.prefetch_map = {}
        #Holds data objects on this device that are needed by tasks that have not yet completed (this includes data in the process of being prefetched).
        self.active_map = {}
        #holds data objects that are currently being used by tasks.
        self.used_map = {}

    def _add(self, node):

        #Increment usage count
        node.priority += 1

        #Lookup ListNode in priority table
        list_node = self.priority_map.get(node.priority, None)

        if list_node is None:
            #Create new list node
            list_node = ListNode(node.priority, DLList())
            self.priority_list[node.priority] = list_node

            if node.priority == 1:
                #Add to the head of the list
                self.priority_list.append(list_node)
            else:
                #Add as the next node after the previous priority
                self.priority_list.insert_after(list_node, self.priority_list[node.priority - 1])

        #Add data node to internal list
        list_node.list.append(node)

    def _remove(self, node, delete=False):

        #Lookup ListNode in priority table
        list_node = self.priority_map.get(node.priority, None)

        assert(list_node is not None)

        success = list_node.list.remove(node)

        if len(list_node.list) == 0:
            #Remove the list node from the priority list

            self.priority_list.remove(list_node)
            del self.priority_map[list_node.priority]

        if delete:
            del self.data_map[node]

        return success


    def _get_evict_target(self):

        #Get least frequently used node

        #Get the first list node in the priority list

        list_node = self.priority_list.head
        data_node = list_node.list.head

        while list_node is not None:

            #Check all data nodes with the same priority
            while data_node is not None:
                data_node = data_node.next

                if data_node.data.get_active(self.device) == 0:
                    break
            
            #Continue search in next priority list
            if data_node.data.get_active(self.device) == 0:
                break

            list_node = list_node.next

        return data_node

    def _start_prefetch_data(self, data):

        data.add_prefetch(self.device)
        data.add_active(self.device)

        if data in self.data_map:
            #This is a prefetch of a data object that is already on the device (or is being prefetched).
            #This means the data is no longer evictable as its about to be in-use by data movement and compute tasks.

            if data.get_active(self.device) == 1:
                #This is the first prefetch of a data object that is already on the device.
                #Update the evictable memory size (as this data object is no longer evictable).
                self.evictable_memory -= data.size
        else:
            #This is a new block, update the used memory size.
            self.used_memory += data.size

        self.prefetch_map[data] = data
        self.active_map[data] = data

        assert(self.used_memory <= self.memory_limit)

    def _stop_prefetch_data(self, data):

        count = data.get_prefetch(self.device)
        data.remove_prefetch(self.device)

        if count == 1:
            del self.prefetch_map[data]

    def _access_data(self, data):

        #NOTE(wlr): The data should already be removed from the evictable list in the prefetching stage.
        #           Any logic here would be a sanity check. I'm removing it for now.

        #node = self.data_map.get(data, None)
        #if node is not None:
        #    #Remove the node from the eviction list while it's in use
        #    success = self.data_list.remove(node)

        self.used_map[data] = data
        data.add_use(self.device)


    def _release_data(self, data):
        node = self.data_map[data]

        active_count = data.get_active(self.device)
        use_count = data.get_use(self.device)

        data.remove_active(self.device)
        data.remove_use(self.device)

        self._add(node)

        if active_count == 1:
            del self.active_map[data]
            self.evictable_memory += data.nbytes

        if use_count == 1:
            del self.used_map[data]

    def _evict_data(self, data):
        node = self.data_map[data]

        assert(data.get_use(self.device) == 0)
        assert(data.get_active(self.device) == 0)

        #Call internal data object evict method
        #This should:
        #  - Backup the data if its not in a SHARED state
        #    (SHARED state means the data has a valid copy on multiple devices. Eviction should never destroy the only remaining copy)
        #  - Mark the data for deletion (this may be done by the CuPy/Python GC)
        data.evict(self.device)

        self._remove(node, delete=True)
        
        self.used_memory -= data.nbytes
        self.evictable_memory -= data.nbytes


    def _evict(self):
        # Get the oldest data object and remove it
        node = self._get_evict_target()
        self._evict_data(node)

class EvictionManager:
    """
    Track usage of data objects on devices. Used to chose which blocks to evict.
    """

    def __init__(self, device, memory_limit):
        self.device = device

        #values in bytes
        self.memory_limit = memory_limit
        self.used_memory = 0
        self.evictable_memory = 0

        self.lock = threading.Condition(threading.Lock())

    def map_data(self, data):
        """
        Called when a data object is mapped to a device.
        """
        with self.lock:
            self._map_data(data)

    def _map_data(self, data):
        """
        Called when a data object is mapped to a device.
        """
        pass

    def _unmap_data(self, data):
        pass

    def unmap_data(self, data):
        """
        Called when a data object is unmapped from a device.
        """
        with self.lock:
            self._unmap_data(self, data)

    def _start_prefetch_data(self, data):
        pass

    def start_prefetch_data(self, data):
        """
        Called when a data object starts a prefetch.
        Updates the used memory size.

        Can update the priority of the data object.
        """
        with self.lock:
            self._start_prefetch_data(data)

    def _stop_prefetch_data(self, data):
        """
        Called when a data object is no longer being prefetched.

        Updates the priority of the data object.
        """
        # TODO(hc): what does it mean?
        pass

    def stop_prefetch_data(self, data):
        """
        Called when a data object is no longer being prefetched.

        Updates the priority of the data object.
        """
        with self.lock:
            self._stop_prefetch_data(data)

    def _access_data(self, data):
        pass


    def access_data(self, data):
        """
        Called when a data object is accessed.

        Can update the priority of the data object.
        Locks the data object (cannot be evicted while in use)
        Updates the evictable memory size.
        """
        with self.lock:
            self._access_data(data)


    def _release_data(self, data):
        pass

    def release_data(self, data):
        """
        Called when a data object is no longer in use.

        Updates the priority of the data object.
        Unlocks the data object (can be evicted)
        Updates the evictable memory size.
        """
        with self.lock:
            self._release_data(data)

    def _evict_data(self, data):
        pass

    def evict_data(self, data):
        """
        Called to evict a specific data object.
        Updates the used memory size and evictable memory size.
        """
        with self.lock:
            self._evict_data(data)

    def _evict(self):
        """
        Called when memory is needed.

        Evicts the data object with the highest priority (based on the policy).
        """
        pass


    def evict(self):
        """
        Called when memory is needed.

        Evicts the data object with the highest priority (based on the policy).
        """
        with self.lock:
            self._evict()    
'''
