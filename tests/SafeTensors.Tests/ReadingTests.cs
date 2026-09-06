using SafeTensors.Tests.Infrastructure;

namespace SafeTensors.Tests;

/// <summary>
/// Reading files produced by the reference implementation, through each of the three
/// storage backends.
/// </summary>
/// <remarks>
/// The fixtures under <c>TestFiles/reference_*</c> come from the Hugging Face
/// <c>safetensors</c> package, not from this library. A round-trip test only proves the
/// writer and the reader agree with each other; these prove they agree with everyone else.
/// See <c>TestFiles/genTests.py</c>.
/// </remarks>
public class ReadingTests
{
    // Chosen in the generator so every value is exact in F16, BF16, F32 and F64 alike.
    private static readonly float[] Floats = [-2.0f, -0.5f, 0.5f, 2.0f];

    private static string Fixture(string name) => Path.Combine(AppContext.BaseDirectory, "TestFiles", name);

    private static SafeTensorFile AllDTypes() => SafeTensorFile.Open(Fixture("reference_all_dtypes.safetensors"));

    [Fact]
    public void Reads_every_dtype_the_reference_implementation_can_write()
    {
        using SafeTensorFile model = AllDTypes();

        Assert.Equal(13, model.Count);

        Assert.Equal(SafeTensorDType.BOOL, model["bool"].DType);
        Assert.Equal(new byte[] { 1, 0, 1, 0 }, model["bool"].ToArray<byte>());

        Assert.Equal(new byte[] { 1, 2, 3, 4 }, model["u8"].ToArray<byte>());
        Assert.Equal(new sbyte[] { -2, -1, 1, 2 }, model["i8"].ToArray<sbyte>());
        Assert.Equal(new short[] { -2, -1, 1, 2 }, model["i16"].ToArray<short>());
        Assert.Equal(new ushort[] { 1, 2, 3, 4 }, model["u16"].ToArray<ushort>());
        Assert.Equal(new[] { -2, -1, 1, 2 }, model["i32"].ToArray<int>());
        Assert.Equal(new uint[] { 1, 2, 3, 4 }, model["u32"].ToArray<uint>());
        Assert.Equal(new[] { -2L, -1L, 1L, 2L }, model["i64"].ToArray<long>());
        Assert.Equal(new[] { 1UL, 2UL, 3UL, 4UL }, model["u64"].ToArray<ulong>());

        Assert.Equal(Floats, model["f32"].ToArray<float>());
        Assert.Equal([-2.0d, -0.5d, 0.5d, 2.0d], model["f64"].ToArray<double>());
    }

    [Fact]
    public void Reads_F16_written_by_the_reference_implementation()
    {
        using SafeTensorFile model = AllDTypes();

        Assert.Equal(SafeTensorDType.F16, model["f16"].DType);
        Assert.Equal(8, model["f16"].ByteLength);
        Assert.Equal(Floats, model["f16"].ToSingleArray());
    }

    [Fact]
    public void Reads_BF16_written_by_the_reference_implementation()
    {
        // BF16 is the dominant weight format for large models, and it is the one dtype
        // most likely to be decoded by a library that never saw a real file of it.
        using SafeTensorFile model = AllDTypes();

        Assert.Equal(SafeTensorDType.BF16, model["bf16"].DType);
        Assert.Equal(8, model["bf16"].ByteLength);
        Assert.Equal(Floats, model["bf16"].ToSingleArray());

        ReadOnlySpan<BFloat16> raw = model["bf16"].AsSpan<BFloat16>();
        Assert.Equal(-2.0f, (float)raw[0]);
        Assert.Equal(2.0f, (float)raw[3]);
    }

    [Fact]
    public void Reads_the_shapes_the_reference_implementation_considers_edge_cases()
    {
        using SafeTensorFile model = SafeTensorFile.Open(Fixture("reference_edge_shapes.safetensors"));

        // A rank-0 scalar holds one element, not zero.
        TensorView scalar = model["scalar"];
        Assert.Equal(0, scalar.Rank);
        Assert.Equal(1, scalar.ElementCount);
        Assert.Equal(42.0f, scalar.ToArray<float>()[0]);

        // A zero dimension means no elements and no bytes, so start equals end.
        TensorView empty = model["empty"];
        Assert.Equal([0, 4], empty.Shape);
        Assert.Equal(0, empty.ElementCount);
        Assert.Equal(0, empty.ByteLength);
        Assert.Empty(empty.ToArray<float>());

        TensorView rank4 = model["rank4"];
        Assert.Equal([2, 3, 4, 5], rank4.Shape);
        Assert.Equal(120, rank4.ElementCount);
        Assert.Equal(119, rank4.AsSpan<int>()[119]);
    }

    [Fact]
    public void Reads_metadata_written_by_the_reference_implementation()
    {
        using SafeTensorFile model = SafeTensorFile.Open(Fixture("reference_metadata.safetensors"));

        Assert.Equal("np", model.Metadata["format"]);
        Assert.Equal("reference", model.Metadata["framework"]);
        Assert.Equal("1200", model.Header.GetMetadata("step"));
        Assert.Null(model.Header.GetMetadata("absent"));

        // __metadata__ is not a tensor.
        Assert.Single(model.Tensors);
        Assert.Equal([2, 2], model["weight"].Shape);
    }

    [Fact]
    public void Reads_the_header_without_mapping_the_file()
    {
        SafeTensorHeader header = SafeTensorFile.ReadHeader(Fixture("reference_all_dtypes.safetensors"));

        Assert.Equal(13, header.Tensors.Count);
        Assert.Equal(header.HeaderSize + 8, header.DataOffset);
        Assert.Equal(SafeTensorDType.BF16, header.Tensors["bf16"].DType);
    }

    [Fact]
    public async Task Reads_the_header_asynchronously()
    {
        using var stream = File.OpenRead(Fixture("reference_all_dtypes.safetensors"));

        SafeTensorHeader header = await SafeTensorFile.ReadHeaderAsync(stream);

        Assert.Equal(13, header.Tensors.Count);
    }

    [Fact]
    public void Reads_from_a_byte_array_without_copying()
    {
        byte[] bytes = File.ReadAllBytes(Fixture("reference_all_dtypes.safetensors"));

        using SafeTensorFile model = SafeTensorFile.Read(bytes);

        Assert.True(model["f32"].IsZeroCopy);
        Assert.Equal(Floats, model["f32"].ToArray<float>());
    }

    [Fact]
    public void Reads_from_a_stream_on_demand()
    {
        using var stream = File.OpenRead(Fixture("reference_all_dtypes.safetensors"));
        using SafeTensorFile model = SafeTensorFile.Read(stream, leaveOpen: true);

        Assert.False(model["f32"].IsZeroCopy);
        Assert.Equal(Floats, model["f32"].ToArray<float>());
        Assert.Equal(new sbyte[] { -2, -1, 1, 2 }, model["i8"].ToArray<sbyte>());
    }

    [Fact]
    public async Task Reads_tensor_bytes_asynchronously_from_a_stream()
    {
        using var stream = File.OpenRead(Fixture("reference_all_dtypes.safetensors"));
        using SafeTensorFile model = SafeTensorFile.Read(stream, leaveOpen: true);

        var buffer = new byte[model["u8"].ByteLength];
        await model["u8"].CopyToAsync(buffer);

        Assert.Equal(new byte[] { 1, 2, 3, 4 }, buffer);
    }

    [Fact]
    public void All_three_backends_return_identical_bytes()
    {
        string path = Fixture("reference_all_dtypes.safetensors");
        byte[] bytes = File.ReadAllBytes(path);

        using SafeTensorFile mapped = SafeTensorFile.Open(path);
        using SafeTensorFile memory = SafeTensorFile.Read(bytes);
        using var stream = File.OpenRead(path);
        using SafeTensorFile streamed = SafeTensorFile.Read(stream, leaveOpen: true);

        foreach (string name in mapped.Names)
        {
            Assert.Equal(mapped[name].ToArray(), memory[name].ToArray());
            Assert.Equal(mapped[name].ToArray(), streamed[name].ToArray());
        }
    }

    [Fact]
    public void Refuses_a_stream_that_cannot_seek()
    {
        using var stream = new NonSeekableStream(File.ReadAllBytes(Fixture("reference_all_dtypes.safetensors")));

        Assert.Throws<ArgumentException>(() => SafeTensorFile.Read(stream));
    }

    [Fact]
    public void Opens_a_file_marked_read_only()
    {
        using TempDirectory directory = Infrastructure.Fixture.NewDirectory();
        string path = directory.File("readonly.safetensors");

        new SafeTensorBuilder()
            .AddTensor("w", new float[] { 1, 2, 3, 4 }, [4])
            .Save(path);

        File.SetAttributes(path, FileAttributes.ReadOnly);

        try
        {
            using SafeTensorFile model = SafeTensorFile.Open(path);
            Assert.Equal(4, model["w"].ElementCount);
        }
        finally
        {
            File.SetAttributes(path, FileAttributes.Normal);
        }
    }

    [Fact]
    public void Opens_a_file_another_reader_already_has_mapped()
    {
        using TempDirectory directory = Infrastructure.Fixture.NewDirectory();
        string path = directory.File("shared.safetensors");

        new SafeTensorBuilder()
            .AddTensor("w", new float[] { 1, 2, 3, 4 }, [4])
            .Save(path);

        using SafeTensorFile first = SafeTensorFile.Open(path);
        using SafeTensorFile second = SafeTensorFile.Open(path);

        Assert.Equal(first["w"].ToArray<float>(), second["w"].ToArray<float>());
    }

    [Fact]
    public void Missing_tensors_report_the_name_they_were_asked_for()
    {
        using SafeTensorFile model = AllDTypes();

        SafeTensorNotFoundException error = Assert.Throws<SafeTensorNotFoundException>(() => model["nope"]);

        Assert.Equal("nope", error.TensorName);
        Assert.False(model.TryGetTensor("nope", out TensorView? missing));
        Assert.Null(missing);
    }

    [Fact]
    public void Using_a_disposed_file_throws_rather_than_reading_freed_pages()
    {
        SafeTensorFile model = AllDTypes();
        TensorView tensor = model["f32"];

        ReadOnlyMemory<float> retained = tensor.AsMemory<float>();
        Assert.Equal(4, retained.Length);

        model.Dispose();

        // A raw span would read unmapped memory here and take the process down. Memory
        // re-checks the mapping on every access, which is the whole reason it exists.
        Assert.Throws<ObjectDisposedException>(() => retained.Span[0]);
        Assert.Throws<ObjectDisposedException>(() => model["f32"]);
    }

    [Fact]
    public void Disposing_twice_is_harmless()
    {
        SafeTensorFile model = AllDTypes();

        model.Dispose();
        model.Dispose();
    }

    private sealed class NonSeekableStream(byte[] data) : MemoryStream(data)
    {
        public override bool CanSeek => false;
    }
}
