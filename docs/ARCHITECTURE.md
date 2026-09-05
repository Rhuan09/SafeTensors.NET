# Architecture

Why things are shaped the way they are. Read this before changing the storage abstraction,
the validation rules, or anything that hands out a span.

## The format, briefly

A SafeTensors file is three parts, in order:

```
[ uint64 little-endian header length ][ that many bytes of UTF-8 JSON ][ raw tensor bytes ]
```

The JSON maps tensor names to `{ dtype, shape, data_offsets }`, where the offsets are byte
positions relative to the start of the data section. An optional `__metadata__` key holds
string key-value pairs. That is the whole format.

Two consequences shape everything here:

1. **Everything descriptive is in the header.** You never need to read the weights to know
   what a file contains, which is why `ReadHeader` exists as a first-class operation rather
   than a shortcut.
2. **The data section is a flat byte range with no structure of its own.** Tensors are
   located purely by the offsets the header claims. If those offsets lie, nothing downstream
   catches it. That is why validation is where it is.

## Three layers

```
SafeTensorFile / ShardedSafeTensorFile     names, lifetime, factories
        |
    TensorView                             one tensor: shape, dtype, accessors
        |
   ITensorDataSource                       bytes from somewhere (internal)
        |
   MemoryMapped / Memory / Stream
```

`ITensorDataSource` is deliberately **internal**. It exposes raw pointers, and making that
a public extension point would commit the package to supporting third-party implementations
of an unsafe contract forever, in exchange for a scenario nobody has asked for. Adding a
source later is a non-breaking change; taking the interface back is not.

The three implementations differ in exactly one interesting way:

| Source | `IsZeroCopy` | Pointer | Notes |
| --- | --- | --- | --- |
| `MemoryMappedDataSource` | yes | yes | The reason the library exists |
| `MemoryDataSource` | yes | no | Managed buffers move; it declines rather than lie |
| `StreamDataSource` | no | no | Seeks and copies per read, serialised on one gate |

## Lifetimes

This is the design decision with the widest blast radius, so it is worth stating plainly.

A memory-mapped span points at pages that `Dispose` unmaps. A `ReadOnlySpan<T>` returned
from a method can escape the scope it was created in, and C# has no way to say "this span
may not outlive that object". So a span used after the file is disposed reads memory the
process no longer owns. That is an `AccessViolationException` — the process dies, and no
`catch` intervenes.

Three options were on the table:

1. **Reference-count the mapping.** Does not work: spans have no destructor, so there is no
   moment at which to decrement.
2. **Never hand out spans.** Throws away the entire point.
3. **Make the safe path the easy one and name the unsafe path.** What we do.

So: `AsSpan<T>()` keeps the name people expect and checks disposal at the moment of the
call. `AsMemory<T>()` goes through a `MemoryManager<T>` that re-checks the mapping on
*every* `.Span` access, turning the crash into `ObjectDisposedException` — that is the
accessor for anything that has to be stored in a field or captured by a lambda.
`ToArray<T>()` copies and owns itself. The pointer accessor is `DangerousGetPointer()`, a
method rather than a property, because the name is the documentation.

The README states the rule in one line: *use a span inside the `using`, use memory or an
array to escape it.*

## Validation

A checkpoint is attacker-controlled input in the ordinary case, not the paranoid one — it
arrives over the network from a model hub. The header's length prefix is a bare `uint64` at
offset zero, read before anything about the file has been verified.

Validation happens once, in `SafeTensorHeader.Parse`, before any `TensorView` exists. By the
time a caller holds a view, the layout is known to be consistent, so the accessors do bounds
checks against a range that is already known to be inside the file.

The rules, and why each one is there:

**Header length cap, checked before allocating.** A hostile file can ask for 16 exabytes.
Without the cap that is an allocation, not an exception.

**Offsets are read through one helper that converts every failure mode.**
`JsonElement.TryGetInt64` *throws* `InvalidOperationException` when the value is not a
number — it does not return `false`. Getting that wrong is how a raw framework exception
ends up escaping a parser that promised to throw `SafeTensorException`.

**Ranges must tile the data section.** Sorted by start offset, each must begin where the
previous ended. Overlap is always an error: two names claiming the same bytes lets one file
hand the same memory to two consumers that each believe they own it. Gaps are an error by
default but can be allowed, because padding to a block boundary is a real thing producers
do and it is harmless — nothing addresses those bytes.

**Element counts are computed `checked`.** `[2^62, 2^62]` wraps to a small positive number
in unchecked arithmetic, which would then agree with a tiny byte range and pass the
size check. The wrap is the attack; `checked` is the fix.

**Duplicate names are rejected.** `JsonDocument` preserves duplicate keys rather than
collapsing them, and different JSON parsers resolve them differently. A file with one name
defined twice at different offsets decodes differently depending on who reads it.

**Shard names cannot escape the model directory.** A `weight_map` entry is a file name
chosen by whoever published the model. `ResolveShardPath` rejects rooted paths and anything
that resolves outside the directory holding the index.

## Writing

The header is built into a `MemoryStream` first, so its length is known before the length
prefix is written. There is no fixed header budget, which means no size at which writing
starts silently producing corrupt files.

`WriteFile` is atomic in the sense that matters: write to a temporary file in the *same
directory*, `Flush(flushToDisk: true)`, then one rename onto the target. Same directory so
the rename stays within one volume; flush first so the rename cannot land before the data
does. `File.Move(overwrite: true)` maps to `MoveFileEx` with `REPLACE_EXISTING` on Windows
and `rename(2)` on Unix, both of which are atomic. On netstandard2.0 the same job is done by
`File.Replace`.

Deleting the target and then moving would leave a window in which neither file exists. For a
library whose job is other people's model checkpoints, that window is not acceptable.

## Not copying, on both sides

`TensorItem` created from an array or a `ReadOnlyMemory<T>` keeps a reference and marshals
the bytes at write time. Staging a model in the builder therefore costs the size of the
header, not the size of the model. The cost is the usual builder contract: the source must
not change before `Save`. `FromSpan` is the exception and copies, because a span cannot be
stored.

On netstandard2.0, `Stream` has no `Write(ReadOnlySpan<byte>)`, so `StreamWrite.Span` copies
through a pooled 128 KB buffer rather than calling `ToArray()`. Calling `ToArray()` there
would undo the whole design in one line.

## Float16 and BFloat16

`Half` does not exist on netstandard2.0, and a tensor library that drops F16 on .NET
Framework and Unity is not much of a tensor library. So `Float16` is a distinct type that is
bit-identical to `Half` and converts to and from it for free on .NET 5 and later.

The software conversion is compiled on **every** target, not just netstandard2.0. It has to
be: a conversion path that only compiles on the framework nobody runs tests on is a
conversion path nobody tests. `NumericsTests` runs it against `Half` across all 65 536 bit
patterns, their float neighbours, and 200 000 pseudo-random floats.

The one deliberate difference from `Half`: NaN collapses to a canonical quiet NaN rather
than a truncated payload, so a truncation can never land on infinity. NaN payloads are
unspecified by IEEE 754, so both are correct; ours cannot produce a wrong *kind* of value.

## Multi-targeting

`netstandard2.0`, `net8.0`, `net10.0`. `net6.0` left support in November 2024 and is not
targeted. netstandard2.0 stays because .NET Framework and Unity consumers are a real part of
the audience for this format, and the polyfill cost is five packages that only that TFM
pulls.

Modern TFMs have **zero** package dependencies, and are marked `IsAotCompatible` and
`IsTrimmable`. The one reflection-based API, `DeserializeMetadata<T>`, carries
`RequiresUnreferencedCode` and `RequiresDynamicCode` so a trimmed app gets a warning at the
call site instead of a failure at runtime.

## The native ABI

`SafeTensors.Native` publishes with Native AOT and `NativeLib=Shared`. The exports are
`[UnmanagedCallersOnly]` statics; handles are `GCHandle`s to a context object holding the
open file.

Two functions return heap strings the caller must free, and `safetensors.h` says which.
Everything else returns a borrowed pointer or a status code. Shape pointers are pinned
copies owned by the handle, freed by `safetensors_close`, because the alternative — making
the caller supply a buffer — doubles the round trips for a few dozen bytes.

Dtype values cross the boundary as integers, so the enum in `SafeTensorDType` and the one in
`safetensors.h` must stay in the same order. That is the sharpest edge in this project and
it is why adding a dtype is a documented four-step change in
[CONTRIBUTING.md](../CONTRIBUTING.md).
