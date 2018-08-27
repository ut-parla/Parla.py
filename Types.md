# Type system for Parla

This describes a type system for values in Parla (and Gergo).

    ● object
    │    (Arrays)
    ├─● Array[T]
    │ ├─● Array[T].Mutable
    │ └─● Array[T].Immutable
    │
    └─● Sized
      │
      │    (Primitives)
      ├─● I[size]    (Signed Integer)
      ├─● UI[size]   (Unsigned Integer)
      ├─● F[size]    (IEEE-like Floating-point)
      │
      │    (Built-ins)
      ├─● Ref[T]     (A reference to a value of type T)
      ├─● Lock       (A lock or mutex)
      │
      │    (Structures)
      ├─● S
      │ ├─● S.Mutable
      │ └─● S.Immutable
      │ .
      │ .
      │ .
      │
      │ ┌─ Array[T] ┄
      └─● StaticArray[T, s₁, ..., sₙ]
        │ ┌─ Array[T].Mutable ┄
        ├─● StaticArray[T, s₁, ..., sₙ].Mutable
        │ ┌─ Array[T].Immutable ┄
        └─● StaticArray[T, s₁, ..., sₙ].Immutable
