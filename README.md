# SafeTensors.NET

[![ci](https://github.com/Rhuan09/SafeTensors.NET/actions/workflows/ci.yml/badge.svg)](https://github.com/Rhuan09/SafeTensors.NET/actions/workflows/ci.yml)
[![NuGet](https://img.shields.io/nuget/v/SafeTensors.NET.svg)](https://www.nuget.org/packages/SafeTensors.NET)
[![Downloads](https://img.shields.io/nuget/dt/SafeTensors.NET.svg)](https://www.nuget.org/packages/SafeTensors.NET)
[![License: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

**Read and write SafeTensors checkpoints from .NET without installing a deep learning
runtime to do it — and without trusting the file that arrived over the network.**

The package is 150 KB with no dependencies on `net8.0` or `net10.0`, nothing native, and
nothing to configure. It does one thing: it gets you the bytes of a tensor, correctly.

That matters when reading the weights is the whole job rather than the first step of
inference — inspecting a checkpoint, validating one in CI, converting between formats,
merging LoRA adapters, or shipping something into Unity where a native ML stack is not on
the table.

A 40 GB checkpoint opens in the time it takes to parse its header. Tensors come back as
`ReadOnlySpan<T>` pointing straight at the mapped pages, so reading a weight matrix costs
what the page fault costs and nothing else. Writing goes the other way with the same
discipline: the builder holds references to your arrays rather than copies of them, and the
file it produces replaces the old one in a single atomic rename.

```csharp
using var model = SafeTensorFile.Open("model.safetensors");

ReadOnlySpan<float> weights = model["model.layers.0.self_attn.q_proj.weight"].AsSpan<float>();
```

## Install

```bash
dotnet add package SafeTensors.NET
```

Targets `netstandard2.0`, `net8.0` and `net10.0`. The netstandard2.0 build carries
polyfills so .NET Framework 4.6.1+ and Unity get the same API, including F16 support.
On `net8.0` and `net10.0` the package has no dependencies at all, and is marked
trim-safe and Native AOT compatible.

<details>
<summary>Installing from GitHub Packages instead</summary>

Every release also goes to this repository's GitHub Packages feed. That feed requires
authentication even for public packages — a GitHub personal access token with `read:packages`
— so nuget.org is the easier route unless you are already authenticating to GitHub for other
packages.

```bash
dotnet nuget add source https://nuget.pkg.github.com/Rhuan09/index.json \
  --name github-rhuan09 \
  --username <your-github-username> \
  --password <a-PAT-with-read:packages> \
  --store-password-in-clear-text
```

</details>

## Reading

Three ways in, depending on where the bytes are:

```csharp
// Memory mapped. Nothing is read until you touch a tensor.
using var fromDisk = SafeTensorFile.Open("model.safetensors");

// Already in memory. Views point into your buffer; still no copy.
using var fromBuffer = SafeTensorFile.Read(bytes);

// A seekable stream — an archive entry, a network stream, a file you do not own.
// Tensor bytes are fetched on demand.
using var fromStream = SafeTensorFile.Read(stream, leaveOpen: true);
```

Then reach for the accessor that matches how long you need the data:

| Accessor | Cost | Valid for |
| --- | --- | --- |
| `AsSpan<T>()` | nothing | the enclosing scope of the file's `using` |
| `AsMemory<T>()` | nothing | anywhere — throws `ObjectDisposedException` after the file closes |
| `ToArray<T>()` | one copy | forever; it owns itself |
| `OpenStream()` | nothing | until you dispose the stream |
| `DangerousGetPointer()` | nothing | until `Dispose`, for native interop and GPU uploads |

### Inspecting without loading

Everything you need to describe a checkpoint lives in its header, so you never have to map
the weights to find out what is in them:

```csharp
SafeTensorHeader header = SafeTensorFile.ReadHeader("llama-70b-00001-of-00015.safetensors");

foreach (TensorMetadata tensor in header.Tensors.Values)
{
    Console.WriteLine($"{tensor.Name}: {tensor.DType} [{string.Join(", ", tensor.Shape)}]");
}
```

### Tensors larger than 2 GiB

`Span<T>` addresses at most `int.MaxValue` elements, and an fp32 embedding matrix passes
that comfortably. Rather than pretend otherwise, the library says so and gives you three
ways through:

```csharp
TensorView big = model["embedding.weight"];

// A window of elements.
ReadOnlySpan<float> chunk = big.AsSpan<float>(elementOffset: 1_000_000, count: 4096);

// A contiguous run of the outermost dimension, as a tensor in its own right. Free:
// row-major layout means those rows are one byte range.
TensorView rows = big.Slice(start: 5000, count: 32);

// Or stream it.
using Stream sequential = big.OpenStream();
```

`big.AsSpan<float>()` on a tensor that size throws `TensorTooLargeException` with a message
naming these, instead of an `OverflowException` from somewhere in the middle of a cast.

### Sharded models

```csharp
using var model = ShardedSafeTensorFile.Open("model.safetensors.index.json");

// Only the shard holding this tensor is opened, and it stays open for the next one.
ReadOnlySpan<float> values = model["model.layers.61.mlp.down_proj.weight"].AsSpan<float>();
```

## Writing

```csharp
new SafeTensorBuilder()
    .WithMetadata("format", "pt")
    .AddTensor("embedding.weight", embeddings, [vocabSize, hiddenSize])
    .AddTensor("layer.0.bias", bias, [hiddenSize])
    .Save("model.safetensors");
```

The builder references your arrays rather than copying them, so staging an entire model
costs no extra memory — and, as with any builder, the arrays must not change before you
call `Save`. The header is sized from the JSON it actually produces, so there is no fixed
budget to overflow.

To rewrite or reshard an existing checkpoint, hand the builder a tensor from another file
and its bytes stream from one to the other:

```csharp
using var source = SafeTensorFile.Open("in.safetensors");

var output = new SafeTensorBuilder();
foreach (TensorView tensor in source.Tensors.Values.Where(t => !t.Name.Contains("lora")))
{
    output.AddTensor(tensor);
}

output.Save("out.safetensors");
```

`Save` writes to a temporary file in the same directory, flushes it through to the device,
and then replaces the target in one rename. A crash at any point leaves either the old file
or a stray temporary — never a valid-looking file with a truncated model inside it.

## Validation

A checkpoint is untrusted input. It usually arrives over the network from a model hub, its
header is attacker-controlled JSON, and its length prefix is a bare 64-bit integer at offset
zero. The reader validates the whole layout before handing out a single byte:

- **Overlapping tensors are rejected**, always. Two names claiming the same bytes is not
  something an honest producer writes, and accepting it lets one file hand the same memory
  to two consumers that each believe they own it.
- **Gaps are rejected** by default; opt in with `AllowNonContiguousData` for padded files.
- **Shapes that overflow are rejected.** `[2^62, 2^62]` wraps to a small positive number in
  unchecked arithmetic, which would then agree with a tiny byte range and pass.
- **Duplicate tensor names are rejected.** JSON has no duplicate-key rule that parsers agree
  on, so a file with one name defined twice decodes differently depending on who reads it.
- **The header length is capped** at 100 MiB by default, before anything is allocated for it.
- **Shard names cannot escape the model directory.** A `weight_map` entry is a file name
  chosen by whoever published the model; `../../../../etc/shadow` does not open.

Everything the library raises derives from `SafeTensorException`, so a `catch` around
`Open` does not have to also anticipate an `InvalidOperationException` from a JSON reader.

```csharp
var options = new SafeTensorReadOptions
{
    MaxHeaderSize = 8 * 1024 * 1024,
    AllowNonContiguousData = false,
    AllowTrailingBytes = false,
};

using var model = SafeTensorFile.Open(path, options);
```

## Lifetimes, and the one thing to know

When a file is memory mapped, `AsSpan<T>()` returns a span pointing at mapped pages. Those
pages are unmapped when the `SafeTensorFile` is disposed, and **a span that outlives the
file reads memory the process no longer owns** — which is an access violation, not an
exception you can catch.

C# cannot express that constraint in the type system, so the library is built so you rarely
need to think about it:

```csharp
// Fine: the span never leaves the scope that owns the file.
using (var model = SafeTensorFile.Open(path))
{
    Process(model["w"].AsSpan<float>());
}

// Also fine: memory re-checks the mapping on every access.
ReadOnlyMemory<float> retained;
using (var model = SafeTensorFile.Open(path))
{
    retained = model["w"].AsMemory<float>();
}
retained.Span[0];   // ObjectDisposedException, not a crash
```

Use a span inside the `using`. Use `AsMemory<T>()` or `ToArray<T>()` to escape it. The
pointer accessor is called `DangerousGetPointer()` for the same reason.

## Data types

`BOOL`, `U8`, `I8`, `I16`, `U16`, `F16`, `BF16`, `I32`, `U32`, `F32`, `F64`, `I64`, `U64`,
`F8_E4M3`, `F8_E5M2`.

`F16` and `BF16` get first-class struct types — `Float16` and `BFloat16` — that are exactly
two bytes and blittable, so `AsSpan<BFloat16>()` casts straight over the file's bytes.
`Float16` converts to and from `System.Half` for free on .NET 5 and later; on netstandard2.0
it uses its own conversion, which is the same code, compiled everywhere and checked against
`Half` over all 65 536 bit patterns in the test suite.

The 8-bit float types have no CLR counterpart. They read as raw bytes and are written
through the explicit-dtype overload.

## Native ABI

Every release ships a Native AOT shared library with a C ABI, so a consumer in C, C++,
Rust or Python gets the weights at their address in the page cache — **with no .NET
runtime installed on the machine**.

Download from [the releases page](https://github.com/Rhuan09/SafeTensors.NET/releases):
`safetensors-win-x64.dll`, `safetensors-linux-x64.so`, `safetensors-osx-arm64.dylib`, and
the `safetensors.h` they were built against. For any other platform, build it yourself —
Native AOT cannot cross-compile, so only the three with a CI runner are prebuilt:

```bash
dotnet publish src/SafeTensors.Native -c Release -r linux-arm64
```

```c
#include "safetensors.h"

safetensors_handle_t model = safetensors_open("model.safetensors");
if (!model) {
    char* error = safetensors_get_last_error();
    fprintf(stderr, "%s\n", error);
    safetensors_free_string(error);          /* the caller owns this string */
    return 1;
}

uint64_t bytes = 0;
const float* weights = safetensors_get_tensor_data_ptr(model, "embedding.weight", &bytes);

safetensors_close(model);                    /* weights is invalid after this */
```

`safetensors.h` states which returns are owned and which are borrowed.

## Benchmarks

`dotnet run -c Release --project benchmarks/SafeTensors.Benchmarks`. A short run on one
Windows 11 laptop, .NET 10 — treat the wall-clock numbers as indicative and the allocation
column as the point:

| | 16 tensors | 1024 tensors | Allocated |
| --- | ---: | ---: | ---: |
| `ReadHeader` | 90 µs | 1.00 ms | header-sized |
| `Open` (mapped) | 130 µs | 1.05 ms | header-sized |
| Sum one tensor via `AsSpan<float>()` | **2.708 µs** | **2.706 µs** | **0 B** |
| Sum one tensor via `ToArray<float>()` | 5.9 µs | 5.6 µs | 16 KB |
| Sum all tensors via spans | 43 µs | 3.0 ms | **43 B** |

Two things worth reading off that table. Reading a tensor takes the same time whether the
file holds 16 of them or 1024, because a view is a byte range and finding it is a dictionary
lookup. And summing four million floats across 1024 tensors allocates 43 bytes total — the
enumerator, and nothing else.

Opening scales with the *header*, not the data. These files are small, so that is what the
first two rows are measuring; the property they demonstrate is that the cost is proportional
to tensor count rather than tensor size.

## Status

The reader, writer, sharded loader and native ABI all work and are covered by the test
suite, which reads files produced by the reference implementation across all thirteen
dtypes it can express, BF16 included.

Not there yet:

- The sub-byte MX float types. The size API is expressed in bits so they can be added
  without a breaking change, but encodings are not guessed at.
- Writing a shard index. Sharded models can be read, not produced.
- Strided slicing of inner dimensions. `Slice` covers the outermost dimension, which is
  contiguous; anything else would be a copy wearing a view's name.
- `System.Numerics.Tensors` interop.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) has the branching model and the PR checklist.
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains why things are shaped the way they
are — read it before proposing a change to the storage abstraction or the validation rules.

```bash
dotnet build
dotnet test
dotnet run -c Release --project benchmarks/SafeTensors.Benchmarks
```

## Licence

[MIT](LICENSE).
