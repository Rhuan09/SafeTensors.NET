using BenchmarkDotNet.Attributes;
using SafeTensors;

namespace SafeTensors.Benchmarks;

/// <summary>
/// Writing costs, and specifically whether staging a model to write it doubles its memory.
/// </summary>
[MemoryDiagnoser]
public class WriteBenchmarks
{
    private float[][] _tensors = null!;
    private string _directory = null!;

    /// <summary>Tensors written per run.</summary>
    [Params(64, 512)]
    public int TensorCount { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _directory = Path.Combine(Path.GetTempPath(), "safetensors-write-bench");
        Directory.CreateDirectory(_directory);

        _tensors = new float[TensorCount][];
        for (int i = 0; i < TensorCount; i++)
        {
            _tensors[i] = new float[4096];
        }
    }

    [GlobalCleanup]
    public void Cleanup() => Directory.Delete(_directory, recursive: true);

    /// <summary>
    /// Staging every tensor in the builder. The allocation column is the one that matters:
    /// the builder holds references rather than copies, so this should not scale with the
    /// total weight size.
    /// </summary>
    [Benchmark]
    public int Stage()
    {
        var builder = new SafeTensorBuilder();
        for (int i = 0; i < _tensors.Length; i++)
        {
            builder.AddTensor($"layer.{i}.weight", _tensors[i], [_tensors[i].Length]);
        }

        return builder.Count;
    }

    /// <summary>Serialising to memory.</summary>
    [Benchmark]
    public int WriteToBytes()
    {
        var builder = new SafeTensorBuilder();
        for (int i = 0; i < _tensors.Length; i++)
        {
            builder.AddTensor($"layer.{i}.weight", _tensors[i], [_tensors[i].Length]);
        }

        return builder.ToByteArray().Length;
    }

    /// <summary>Writing to disk, including the flush and the atomic replace.</summary>
    [Benchmark]
    public void SaveToDisk()
    {
        var builder = new SafeTensorBuilder();
        for (int i = 0; i < _tensors.Length; i++)
        {
            builder.AddTensor($"layer.{i}.weight", _tensors[i], [_tensors[i].Length]);
        }

        builder.Save(Path.Combine(_directory, "bench.safetensors"));
    }
}
