using SafeTensors.Tests.Infrastructure;

namespace SafeTensors.Tests;

/// <summary>
/// Reading files produced by the reference PyTorch implementation, through each of the
/// three storage backends.
/// </summary>
public class ReadingTests
{
    private static string Fixture(string name) => Path.Combine(AppContext.BaseDirectory, "TestFiles", name);

    [Fact]
    public void Reads_a_file_written_by_pytorch()
    {
        using SafeTensorFile model = SafeTensorFile.Open(Fixture("basic_model.safetensors"));

        Assert.Equal(2, model.Count);

        TensorView embedding = model["embedding"];
        Assert.Equal(SafeTensorDType.F32, embedding.DType);
        Assert.Equal(new long[] { 2, 2 }, embedding.Shape);
        Assert.Equal(16, embedding.ByteLength);
        Assert.All(embedding.AsSpan<float>().ToArray(), value => Assert.Equal(0f, value));

        TensorView attention = model["attention"];
        Assert.Equal(SafeTensorDType.I8, attention.DType);
        Assert.Equal(new sbyte[] { 1, 2, 3, 4, 5, 6 }, attention.ToArray<sbyte>());
    }

    [Fact]
    public void Reads_metadata_written_by_pytorch()
    {
        using SafeTensorFile model = SafeTensorFile.Open(Fixture("with_metadata.safetensors"));

        Assert.Equal("value1", model.Metadata["key1"]);
        Assert.Equal("value2", model.Metadata["key2"]);
        Assert.Equal("value1", model.Header.GetMetadata("key1"));
        Assert.Null(model.Header.GetMetadata("absent"));
    }

    [Fact]
    public void Reads_the_header_without_mapping_the_file()
    {
        SafeTensorHeader header = SafeTensorFile.ReadHeader(Fixture("basic_model.safetensors"));

        Assert.Equal(2, header.Tensors.Count);
        Assert.Equal(header.HeaderSize + 8, header.DataOffset);
        Assert.Equal(SafeTensorDType.F32, header.Tensors["embedding"].DType);
    }

    [Fact]
    public async Task Reads_the_header_asynchronously()
    {
        using var stream = File.OpenRead(Fixture("basic_model.safetensors"));

        SafeTensorHeader header = await SafeTensorFile.ReadHeaderAsync(stream);

        Assert.Equal(2, header.Tensors.Count);
    }

    [Fact]
    public void Reads_from_a_byte_array_without_copying()
    {
        byte[] bytes = File.ReadAllBytes(Fixture("basic_model.safetensors"));

        using SafeTensorFile model = SafeTensorFile.Read(bytes);

        Assert.True(model["embedding"].IsZeroCopy);
        Assert.Equal(16, model["embedding"].ByteLength);
    }

    [Fact]
    public void Reads_from_a_stream_on_demand()
    {
        using var stream = File.OpenRead(Fixture("basic_model.safetensors"));
        using SafeTensorFile model = SafeTensorFile.Read(stream, leaveOpen: true);

        Assert.False(model["embedding"].IsZeroCopy);
        Assert.Equal(new sbyte[] { 1, 2, 3, 4, 5, 6 }, model["attention"].ToArray<sbyte>());
    }

    [Fact]
    public async Task Reads_tensor_bytes_asynchronously_from_a_stream()
    {
        using var stream = File.OpenRead(Fixture("basic_model.safetensors"));
        using SafeTensorFile model = SafeTensorFile.Read(stream, leaveOpen: true);

        var buffer = new byte[model["attention"].ByteLength];
        await model["attention"].CopyToAsync(buffer);

        Assert.Equal(new byte[] { 1, 2, 3, 4, 5, 6 }, buffer);
    }

    [Fact]
    public void Refuses_a_stream_that_cannot_seek()
    {
        using var stream = new NonSeekableStream(File.ReadAllBytes(Fixture("basic_model.safetensors")));

        Assert.Throws<ArgumentException>(() => SafeTensorFile.Read(stream));
    }

    [Fact]
    public void Opens_a_file_marked_read_only()
    {
        using TempDirectory directory = Infrastructure.Fixture.NewDirectory();
        string path = directory.File("readonly.safetensors");

        new SafeTensorBuilder()
            .AddTensor("w", new float[] { 1, 2, 3, 4 }, new long[] { 4 })
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
            .AddTensor("w", new float[] { 1, 2, 3, 4 }, new long[] { 4 })
            .Save(path);

        using SafeTensorFile first = SafeTensorFile.Open(path);
        using SafeTensorFile second = SafeTensorFile.Open(path);

        Assert.Equal(first["w"].ToArray<float>(), second["w"].ToArray<float>());
    }

    [Fact]
    public void Missing_tensors_report_the_name_they_were_asked_for()
    {
        using SafeTensorFile model = SafeTensorFile.Open(Fixture("basic_model.safetensors"));

        SafeTensorNotFoundException error = Assert.Throws<SafeTensorNotFoundException>(() => model["nope"]);

        Assert.Equal("nope", error.TensorName);
        Assert.False(model.TryGetTensor("nope", out TensorView? missing));
        Assert.Null(missing);
    }

    [Fact]
    public void Using_a_disposed_file_throws_rather_than_reading_freed_pages()
    {
        SafeTensorFile model = SafeTensorFile.Open(Fixture("basic_model.safetensors"));
        TensorView tensor = model["embedding"];

        ReadOnlyMemory<float> retained = tensor.AsMemory<float>();
        Assert.Equal(4, retained.Length);

        model.Dispose();

        // A raw span would read unmapped memory here and take the process down. Memory
        // re-checks the mapping on every access, which is the whole reason it exists.
        Assert.Throws<ObjectDisposedException>(() => retained.Span[0]);
        Assert.Throws<ObjectDisposedException>(() => model["embedding"]);
    }

    [Fact]
    public void Disposing_twice_is_harmless()
    {
        SafeTensorFile model = SafeTensorFile.Open(Fixture("basic_model.safetensors"));

        model.Dispose();
        model.Dispose();
    }

    private sealed class NonSeekableStream(byte[] data) : MemoryStream(data)
    {
        public override bool CanSeek => false;
    }
}
