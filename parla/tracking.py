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

from typing import TypedDict, Dict

class DataNode:
    """
    A node containing a data object (PArray) on a specific device. 
    Used in the linked lists in the EvictionManager
    """
    def __init__(self, data, device, priority=0):
        self.data = data
        self.device = device
        self.priority = priority

        self.next = None
        self.prev = None

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


class DLList:
    """
    A doubly linked list used in the EvictionManager.
    """
    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"DLList({self.head}, {self.tail})"

    def append(self, node):
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            node.prev = self.tail
            self.tail = node

        self.length += 1

    def remove(self, node):
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
        if node.prev is not None:
            node.prev.next = new_node
            new_node.prev = node.prev
        else:
            self.head = new_node
        node.prev = new_node
        new_node.next = node

        self.length += 1

    def insert_after(self, node, new_node):
        if node.next is not None:
            node.next.prev = new_node
            new_node.next = node.next
        else:
            self.tail = new_node
        node.next = new_node
        new_node.prev = node

        self.length += 1

    def __len__(self):
        return self.length


class DataMapType(TypedDict):
    """
    Track information of data instance in a device
    """
    state: str
    ref_count: int


#class LRUManager(EvictionManager):
class LRUManager:
    """
    LRU policy for garbage collecting.
    It mantains a list of the zero-referenced data objects for each device.
    The head of the list is the target task to be evicted
    and the tail of the list is the data used most recently.
    """

    def __init__(self, device, memory_limit):
        super().__init__(device, memory_limit)
        # A list containig zero-reference data objects in a specified device.
        self.zr_data_list = DLList()
        # A dictionary containing all data information on a device.
        self.data_map = Dict[DataMapType]
        # A lock for guarding a reference count.
        self.ref_count_lock = threading.Condition(threading.Lock())

        #Note(wlr): These tracking dictionaries are optional, I just think it's interesting to track.
        #Holds data objects on this device that are being prefetched.
        #XXX(hc): This might be necessary as data being prefetched cannot be used yet but it can avoid
        # unnecessary future data prefetching or move.
        self.prefetch_map = {}
        #Holds data objects on this device that are needed by tasks that have not yet completed (this includes data in the process of being prefetched).
        self.active_map = {}
        #holds data objects that are currently being used by tasks.
        self.used_map = {}

    def _increase_ref_count(self, data):
        with self.ref_count_lock:
            assert(self.data_map[data.ID]["ref_count"] >= 0)
            self.data_map[data.ID]["ref_count"] += 1

    def _decrease_ref_count(self, data):
        with self.ref_count_lock:
            self.data_map[data.ID]["ref_count"] -= 1
            assert(self.data_map[data.ID]["ref_count"] >= 0)

    def _start_prefetch_data(self, data):
        if data.ID in self.data_map:
            #This is a prefetch of a data object that is already on the device (or is being prefetched).
            #This means the data is no longer evictable as its about to be in-use by data movement and compute tasks.
            #Remove it from the evictable list.

            # TODO(hc): but if a data movement task will be executed after a very long time, that also can be evictable.
            #           if memory is full and any task cannot proceed, we can still evict one of data that was prefetched.
            #           but this is very rare case and I am gonna leave it as the future work.
            success = self.zr_data_list.remove(data)

            if success:
                #This is the first prefetch of a data object that is already on the device.
                #Update the evictable memory size (as this data object is no longer evictable).
                self.evictable_memory -= data.size
            self._increase_ref_count(data)
        else:
            self.data_map[data.ID] = {"state" : "Reserved", "ref_count" : 1}
            #This is a new block, update the used memory size.
            self.used_memory += data.size
        self.prefetch_map[data] = data
        self.active_map[data] = data

        assert(self.used_memory <= self.memory_limit)

    '''
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

        if active_count == 1:
            del self.active_map[data]

            #If the data object is no longer needed by any already prefetched tasks, it can be evicted.
            self.data_list.append(node)
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

        self.data_list.remove(node)
        del self.data_map[data]

        self.used_memory -= data.nbytes
        self.evictable_memory -= data.nbytes

    def _evict(self):
        # Get the oldest data object
        # Because we append after use this is at the front of the list

        node = self.data_list.head
        self._evict_data(node)
    '''

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
