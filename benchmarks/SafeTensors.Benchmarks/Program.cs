using BenchmarkDotNet.Running;

// dotnet run -c Release --project benchmarks/SafeTensors.Benchmarks
// dotnet run -c Release --project benchmarks/SafeTensors.Benchmarks -- --filter "*Read*"
BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);

/// <summary>Entry point marker for BenchmarkSwitcher.</summary>
public partial class Program;
