# Type system for Parla

This describes a type system for values in Parla (and Gergo).

#### Atomic Types

This section describes some types that we could support if needed, but are not required for straight-forward parallel programming in Parla.

These types provide atomicity guarantees only within a single device.
If multiple devices share access to a single memory, then atomicity is only guaranteed if operations only happen from one of those devices.
If multiple devices perform operations, the state of these values become undefined and operations on them are undefined (within the scope of the value).

```
    ● Lock       (A lock or mutex)
```

**Atomic primitive types:**
Primitive-like types which provide atomic update and access semantics.
```
    ● AtomicBoolean    (Atomic Boolean)
    ● AtomicI[size]    (Atomic Signed Integer)
    ● AtomicUI[size]   (Atomic Unsigned Integer)
    ● AtomicF[size]    (Atomic IEEE-like Floating-point)
```

## Programming Interfaces

Parlas types each provide an interface of operators and methods (which are not different other than syntax).
Here we will describe the interfaces for the types above.


### Lock Type

`Lock` is a sized type which provides a mutex interface.
```python
self : Lock
    self.lock()                   # (Acquire)
    self.unlock()                 # (Release)
    self.try_lock() : Boolean     # (Non-blocking acquire)
    with self: …                  # (Synchronized block on self)
```

### Atomic Primitive Types

The atomic primitive types implement interfaces which are different from their non-atomic counter parts.
The operations are a subset of the simple [atomic operations in Java](https://docs.oracle.com/javase/10/docs/api/java/util/concurrent/atomic/AtomicInteger.html).
The operations provide several different operations for ordering semantics, and basic atomic read-update-write operations (such as, compare-and-swap and increment-and-get).

**TODO: Specify the exact operations if we need this.**
