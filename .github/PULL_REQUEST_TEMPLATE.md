## What and why

<!-- What changes, and what problem it solves. Link an issue if there is one. -->

## Testing

- [ ] `dotnet test` passes on both target frameworks
- [ ] New behaviour is covered by a test, or the bug fix has a test that failed before it

<!-- If this touches header parsing, say which malformed inputs you tried. -->

## Checklist

- [ ] No new NuGet dependency on `net8.0`/`net10.0`, or it is justified above
- [ ] Validation defaults were not loosened (new leniency is an opt-in `SafeTensorReadOptions` property)
- [ ] Nothing but `SafeTensorException` can escape the parser
- [ ] `SafeTensorDType` and the enum in `safetensors.h` are still in the same order

## Performance impact

<!-- Required if this touches AsSpan, AsMemory, Slice, the data sources, or the writer's
     hot path. Paste the relevant BenchmarkDotNet rows, before and after. The allocation
     column matters more than the wall clock. Write "none" if the change cannot affect it. -->

## Safety impact

<!-- Required if this touches lifetimes, pointers, spans over mapped memory, or path
     resolution. State plainly what a caller can now do that they could not before, and
     what happens if they get it wrong. Write "none" if there is genuinely no change. -->
