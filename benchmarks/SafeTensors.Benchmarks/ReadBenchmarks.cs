using BenchmarkDotNet.Attributes;
using SafeTensors;

namespace SafeTensors.Benchmarks;

/// <summary>
/// What each way of getting at a tensor actually costs.
/// </summary>
/// <remarks>
/// The point of these is not a number to put on a badge. It is to keep the claim honest:
/// if opening a file ever starts scaling with the file's size rather than its header's,
/// or if a span access starts allocating, this is where it shows up.
/// </remarks>
[MemoryDiagnoser]
public class ReadBenchmarks
{
    private string _path = null!;
    private byte[] _bytes = null!;
    private SafeTensorFile _mapped = null!;

    /// <summary>Tensors in the file. A real checkpoint has thousands.</summary>
    [Params(16, 1024)]
    public int TensorCount { get; set; }

    /// <summary>Elements per tensor.</summary>
    [Params(4096)]
    public int ElementsPerTensor { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _path = Path.Combine(Path.GetTempPath(), $"safetensors-bench-{TensorCount}-{ElementsPerTensor}.safetensors");

        var builder = new SafeTensorBuilder();
        for (int i = 0; i < TensorCount; i++)
        {
            var values = new float[ElementsPerTensor];
            for (int j = 0; j < values.Length; j++)
            {
                values[j] = j;
            }

            builder.AddTensor($"model.layers.{i}.self_attn.q_proj.weight", values, [ElementsPerTensor]);
        }

        builder.Save(_path);

        _bytes = File.ReadAllBytes(_path);
        _mapped = SafeTensorFile.Open(_path);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _mapped.Dispose();
        File.Delete(_path);
    }

    /// <summary>Parsing the header alone, with no mapping.</summary>
    [Benchmark(Baseline = true)]
    public int ReadHeaderOnly() => SafeTensorFile.ReadHeader(_path).Tensors.Count;

    /// <summary>Opening and mapping. Should track the header, not the data.</summary>
    [Benchmark]
    public int OpenMapped()
    {
        using SafeTensorFile file = SafeTensorFile.Open(_path);
        return file.Count;
    }

    /// <summary>Reading from a buffer already in memory.</summary>
    [Benchmark]
    public int ReadFromMemory()
    {
        using SafeTensorFile file = SafeTensorFile.Read(_bytes);
        return file.Count;
    }

    /// <summary>Summing one tensor through a zero-copy span. Should not allocate.</summary>
    [Benchmark]
    public float SumViaSpan()
    {
        ReadOnlySpan<float> values = _mapped["model.layers.0.self_attn.q_proj.weight"].AsSpan<float>();

        float total = 0;
        for (int i = 0; i < values.Length; i++)
        {
            total += values[i];
        }

        return total;
    }

    /// <summary>The same sum through a copy, for the size of the difference.</summary>
    [Benchmark]
    public float SumViaCopy()
    {
        float[] values = _mapped["model.layers.0.self_attn.q_proj.weight"].ToArray<float>();

        float total = 0;
        for (int i = 0; i < values.Length; i++)
        {
            total += values[i];
        }

        return total;
    }

    /// <summary>Touching every tensor, the shape of a real model load.</summary>
    [Benchmark]
    public float SumAllTensors()
    {
        float total = 0;
        foreach (TensorView tensor in _mapped.Tensors.Values)
        {
            ReadOnlySpan<float> values = tensor.AsSpan<float>();
            for (int i = 0; i < values.Length; i++)
            {
                total += values[i];
            }
        }

        return total;
    }
}
