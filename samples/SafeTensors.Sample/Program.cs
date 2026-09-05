using System.Diagnostics;
using SafeTensors;

// A tour of the library against a file it writes itself, so it runs anywhere with no
// model download. Every section prints what it did.

string workingDirectory = Path.Combine(Path.GetTempPath(), "safetensors-sample");
Directory.CreateDirectory(workingDirectory);
string modelPath = Path.Combine(workingDirectory, "demo.safetensors");

Console.WriteLine($"Working in {workingDirectory}");
Console.WriteLine();

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------
Console.WriteLine("== Writing ==");

float[] embedding = new float[256 * 8];
for (int i = 0; i < embedding.Length; i++)
{
    embedding[i] = i * 0.001f;
}

new SafeTensorBuilder()
    .WithMetadata("format", "pt")
    .WithMetadata("produced_by", "SafeTensors.NET sample")
    .AddTensor("embedding.weight", embedding, [256, 8])
    .AddTensor("layer.0.bias", new[] { 0.1f, 0.2f, 0.3f, 0.4f }, [4])
    .AddTensor("layer.0.weight", new[] { (BFloat16)1.5f, (BFloat16)(-0.25f) }, [2])
    .Save(modelPath);

Console.WriteLine($"Wrote {new FileInfo(modelPath).Length:N0} bytes to demo.safetensors");
Console.WriteLine();

// ---------------------------------------------------------------------------
// Header-only inspection
// ---------------------------------------------------------------------------
Console.WriteLine("== Header only ==");

var stopwatch = Stopwatch.StartNew();
SafeTensorHeader header = SafeTensorFile.ReadHeader(modelPath);
stopwatch.Stop();

Console.WriteLine($"Read {header.Tensors.Count} tensor entries in {stopwatch.Elapsed.TotalMilliseconds:F2} ms " +
                  "without mapping a single byte of weights");

foreach (TensorMetadata tensor in header.Tensors.Values.OrderBy(t => t.Name, StringComparer.Ordinal))
{
    Console.WriteLine($"  {tensor.Name,-20} {tensor.DType,-6} [{string.Join(", ", tensor.Shape)}]  {tensor.ByteLength:N0} bytes");
}

Console.WriteLine();

// ---------------------------------------------------------------------------
// Zero-copy reading
// ---------------------------------------------------------------------------
Console.WriteLine("== Reading ==");

using (SafeTensorFile model = SafeTensorFile.Open(modelPath))
{
    Console.WriteLine($"metadata: {string.Join(", ", model.Metadata.Select(m => $"{m.Key}={m.Value}"))}");

    TensorView weights = model["embedding.weight"];

    // No copy: this span points at the file's pages in the OS page cache.
    ReadOnlySpan<float> values = weights.AsSpan<float>();
    Console.WriteLine($"embedding.weight zero-copy: {weights.IsZeroCopy}, first four = " +
                      $"{values[0]}, {values[1]}, {values[2]}, {values[3]}");

    // Slicing the outermost dimension is also free: rows are contiguous.
    TensorView rows = weights.Slice(10, 2);
    Console.WriteLine($"rows 10..12 -> shape [{string.Join(", ", rows.Shape)}], " +
                      $"first value {rows.AsSpan<float>()[0]}");

    // BF16 converts on demand.
    Console.WriteLine($"layer.0.weight as float: {string.Join(", ", model["layer.0.weight"].ToSingleArray())}");

    unsafe
    {
        // The handoff point for native interop and GPU uploads. Valid until Dispose.
        void* pointer = weights.DangerousGetPointer();
        Console.WriteLine($"raw pointer: 0x{(nint)pointer:X}");
    }
}

Console.WriteLine();

// ---------------------------------------------------------------------------
// Sharded models
// ---------------------------------------------------------------------------
Console.WriteLine("== Sharded model ==");

string shardDirectory = Path.Combine(workingDirectory, "sharded");
Directory.CreateDirectory(shardDirectory);

new SafeTensorBuilder()
    .AddTensor("layer.0.weight", new[] { 1f, 2f }, [2])
    .Save(Path.Combine(shardDirectory, "model-00001-of-00002.safetensors"));

new SafeTensorBuilder()
    .AddTensor("layer.1.weight", new[] { 3f, 4f }, [2])
    .Save(Path.Combine(shardDirectory, "model-00002-of-00002.safetensors"));

string indexPath = Path.Combine(shardDirectory, "model.safetensors.index.json");
File.WriteAllText(indexPath, """
    {
      "metadata": { "total_size": "16" },
      "weight_map": {
        "layer.0.weight": "model-00001-of-00002.safetensors",
        "layer.1.weight": "model-00002-of-00002.safetensors"
      }
    }
    """);

using (ShardedSafeTensorFile sharded = ShardedSafeTensorFile.Open(indexPath))
{
    // Only the shard holding this tensor is opened.
    float[] second = sharded["layer.1.weight"].ToArray<float>();
    Console.WriteLine($"layer.1.weight from shard 2: {string.Join(", ", second)}");
}

Console.WriteLine();

// ---------------------------------------------------------------------------
// What a bad file gets you
// ---------------------------------------------------------------------------
Console.WriteLine("== Validation ==");

// Two tensors claiming the same bytes. A reader that accepts this hands the same memory
// to two names.
byte[] overlapping = BuildRawFile(
    """{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]},"b":{"dtype":"U8","shape":[8],"data_offsets":[4,12]}}""",
    dataBytes: 16);

try
{
    using SafeTensorFile _ = SafeTensorFile.Read(overlapping);
    Console.WriteLine("accepted (it should not have been)");
}
catch (SafeTensorValidationException error)
{
    Console.WriteLine($"rejected: {error.Message}");
}

Console.WriteLine();
Console.WriteLine("Done.");

static byte[] BuildRawFile(string headerJson, int dataBytes)
{
    byte[] json = System.Text.Encoding.UTF8.GetBytes(headerJson);
    var buffer = new MemoryStream();
    buffer.Write(BitConverter.GetBytes((ulong)json.Length));
    buffer.Write(json);
    buffer.Write(new byte[dataBytes]);
    return buffer.ToArray();
}
