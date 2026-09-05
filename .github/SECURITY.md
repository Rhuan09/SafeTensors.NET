# Security

## Reporting

Report vulnerabilities through GitHub's private advisory form on this repository
(**Security → Report a vulnerability**). Please do not open a public issue for anything that
lets a crafted file do something it should not.

## Threat model

The input this library is designed against is **a checkpoint file you did not produce**.
That is the ordinary case, not the paranoid one: model weights are downloaded from public
hubs, and both the JSON header and the shard index are chosen entirely by whoever published
them.

In scope, and treated as bugs:

- A crafted header that causes an out-of-bounds read, a wild pointer, or an unbounded
  allocation before validation.
- Tensor byte ranges that overlap, run past the end of the file, or disagree with the shape
  and dtype, being accepted.
- A shard index that causes a file outside the model directory to be opened.
- An exception escaping the parser that does not derive from `SafeTensorException`, if it
  indicates a validation path that was not reached.
- Integer overflow in offset, shape or length arithmetic that turns into a memory-safety
  problem.

Out of scope:

- **Using a span after its file is disposed.** This is a documented API contract, not a
  vulnerability — see the lifetimes section of the README and
  [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md). `AsMemory<T>()` exists for code that
  needs the data to outlive the `using`.
- **`DangerousGetPointer()` misuse.** The name is the warning.
- A file that is merely wrong rather than hostile being rejected. That is a compatibility
  bug — open a normal issue with the file, or with the code that produced it.
- Resource exhaustion from a file the caller chose to open with a raised
  `SafeTensorReadOptions.MaxHeaderSize`.

## What this library does not do

It reads and writes a container format. It does not execute anything, and the format has no
code path by design — that is the point of SafeTensors over pickle. Validating that the
*weights* are what a model expects is the caller's job.
