using System.Text;
using SafeTensors.Tests.Infrastructure;

namespace SafeTensors.Tests;

/// <summary>
/// Writing files, and reading back what was written.
/// </summary>
public class WritingTests
{
    [Fact]
    public void Round_trips_every_dtype_that_has_a_clr_type()
    {
        byte[] file = new SafeTensorBuilder()
            .AddTensor("bool", new[] { true, false, true, true }, [4])
            .AddTensor("u8", new byte[] { 1, 2, 3 }, [3])
            .AddTensor("i8", new sbyte[] { -1, 2, -3 }, [3])
            .AddTensor("i16", new short[] { -300, 300 }, [2])
            .AddTensor("u16", new ushort[] { 1, 65535 }, [2])
            .AddTensor("f16", new[] { (Float16)1.5f, (Float16)(-2.25f) }, [2])
            .AddTensor("bf16", new[] { (BFloat16)1.5f, (BFloat16)(-2.25f) }, [2])
            .AddTensor("i32", new[] { -1, 2, -3 }, [3])
            .AddTensor("u32", new uint[] { 1, 4294967295 }, [2])
            .AddTensor("f32", new[] { 0.1f, -0.5f }, [2])
            .AddTensor("f64", new[] { 0.1d, -0.5d }, [2])
            .AddTensor("i64", new[] { -1L, 2L }, [2])
            .AddTensor("u64", new[] { 1UL, 18446744073709551615UL }, [2])
            .ToByteArray();

        using SafeTensorFile model = SafeTensorFile.Read(file);

        Assert.Equal(13, model.Count);
        Assert.Equal(SafeTensorDType.BOOL, model["bool"].DType);
        Assert.Equal(new byte[] { 1, 2, 3 }, model["u8"].ToArray<byte>());
        Assert.Equal(new sbyte[] { -1, 2, -3 }, model["i8"].ToArray<sbyte>());
        Assert.Equal(new short[] { -300, 300 }, model["i16"].ToArray<short>());
        Assert.Equal(new ushort[] { 1, 65535 }, model["u16"].ToArray<ushort>());
        Assert.Equal(1.5f, (float)model["f16"].AsSpan<Float16>()[0]);
        Assert.Equal(1.5f, (float)model["bf16"].AsSpan<BFloat16>()[0]);
        Assert.Equal(new[] { -1, 2, -3 }, model["i32"].ToArray<int>());
        Assert.Equal(new uint[] { 1, 4294967295 }, model["u32"].ToArray<uint>());
        Assert.Equal(new[] { 0.1f, -0.5f }, model["f32"].ToArray<float>());
        Assert.Equal(new[] { 0.1d, -0.5d }, model["f64"].ToArray<double>());
        Assert.Equal(new[] { -1L, 2L }, model["i64"].ToArray<long>());
        Assert.Equal(new[] { 1UL, 18446744073709551615UL }, model["u64"].ToArray<ulong>());
    }

    [Fact]
    public void Aligns_tensor_data_to_eight_bytes_by_default()
    {
        byte[] file = new SafeTensorBuilder()
            .AddTensor("a", new float[] { 1, 2 }, [2])
            .ToByteArray();

        using SafeTensorFile model = SafeTensorFile.Read(file);

        Assert.Equal(0, model.Header.DataOffset % 8);
    }

    [Fact]
    public void Pads_the_header_with_spaces_so_it_stays_valid_json()
    {
        byte[] file = new SafeTensorBuilder()
            .AddTensor("a", new float[] { 1, 2 }, [2])
            .ToByteArray();

        long headerSize = BitConverter.ToInt64(file, 0);
        string header = Encoding.UTF8.GetString(file, 8, (int)headerSize);

        Assert.Equal('}', header.TrimEnd().Last());
        Assert.All(header.Substring(header.LastIndexOf('}') + 1), c => Assert.Equal(' ', c));
    }

    [Fact]
    public void Writes_a_header_far_larger_than_any_fixed_buffer()
    {
        var builder = new SafeTensorBuilder();
        for (int i = 0; i < 4000; i++)
        {
            builder.AddTensor($"model.layers.{i}.self_attn.q_proj.weight", new byte[] { 1 }, [1]);
        }

        byte[] file = builder.ToByteArray();
        using SafeTensorFile model = SafeTensorFile.Read(file);

        Assert.Equal(4000, model.Count);
        Assert.True(model.Header.HeaderSize > 100_000);
    }

    [Fact]
    public void Refuses_two_tensors_with_the_same_name()
    {
        var builder = new SafeTensorBuilder()
            .AddTensor("w", new float[] { 1 }, [1])
            .AddTensor("w", new float[] { 2 }, [1]);

        SafeTensorValidationException error =
            Assert.Throws<SafeTensorValidationException>(() => builder.ToByteArray());

        Assert.Contains("twice", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Refuses_the_reserved_metadata_name()
    {
        Assert.Throws<ArgumentException>(
            () => new SafeTensorBuilder().AddTensor("__metadata__", new float[] { 1 }, [1]));
    }

    [Fact]
    public void Refuses_data_that_does_not_match_the_shape()
    {
        Assert.Throws<SafeTensorValidationException>(
            () => new SafeTensorBuilder().AddTensor("w", new float[] { 1, 2, 3 }, [4]));
    }

    [Fact]
    public void Writes_metadata()
    {
        byte[] file = new SafeTensorBuilder()
            .WithMetadata("format", "pt")
            .WithMetadata("author", "tests")
            .AddTensor("w", new float[] { 1 }, [1])
            .ToByteArray();

        using SafeTensorFile model = SafeTensorFile.Read(file);

        Assert.Equal("pt", model.Metadata["format"]);
        Assert.Equal("tests", model.Metadata["author"]);
    }

    [Fact]
    public void Replacing_a_file_never_leaves_the_target_missing_or_truncated()
    {
        using TempDirectory directory = Infrastructure.Fixture.NewDirectory();
        string path = directory.File("model.safetensors");

        new SafeTensorBuilder().AddTensor("v1", new float[] { 1 }, [1]).Save(path);
        long firstLength = new FileInfo(path).Length;

        new SafeTensorBuilder().AddTensor("v2", new float[] { 1, 2, 3, 4 }, [4]).Save(path);

        using SafeTensorFile model = SafeTensorFile.Open(path);

        Assert.True(model.Contains("v2"));
        Assert.False(model.Contains("v1"));
        Assert.NotEqual(firstLength, new FileInfo(path).Length);
    }

    [Fact]
    public void A_failed_write_leaves_no_temporary_file_behind()
    {
        using TempDirectory directory = Infrastructure.Fixture.NewDirectory();
        string path = directory.File("model.safetensors");

        var exploding = new TensorItem(
            "w",
            SafeTensorDType.F32,
            [4],
            byteLength: 16,
            writer: _ => throw new InvalidOperationException("source went away"));

        Assert.Throws<InvalidOperationException>(
            () => new SafeTensorBuilder().AddTensor(exploding).Save(path));

        Assert.False(File.Exists(path));
        Assert.Empty(Directory.GetFiles(directory.Path));
    }

    [Fact]
    public void A_failed_write_leaves_the_previous_file_intact()
    {
        using TempDirectory directory = Infrastructure.Fixture.NewDirectory();
        string path = directory.File("model.safetensors");

        new SafeTensorBuilder().AddTensor("good", new float[] { 1 }, [1]).Save(path);

        var exploding = new TensorItem(
            "w",
            SafeTensorDType.F32,
            [4],
            byteLength: 16,
            writer: _ => throw new InvalidOperationException("source went away"));

        Assert.Throws<InvalidOperationException>(
            () => new SafeTensorBuilder().AddTensor(exploding).Save(path));

        using SafeTensorFile model = SafeTensorFile.Open(path);
        Assert.True(model.Contains("good"));
    }

    [Fact]
    public void Copies_a_tensor_from_one_file_into_another_without_materialising_it()
    {
        using TempDirectory directory = Infrastructure.Fixture.NewDirectory();
        string source = directory.File("source.safetensors");
        string target = directory.File("target.safetensors");

        new SafeTensorBuilder()
            .AddTensor("keep", new float[] { 1, 2, 3, 4 }, [2, 2])
            .AddTensor("drop", new float[] { 9, 9 }, [2])
            .Save(source);

        using (SafeTensorFile input = SafeTensorFile.Open(source))
        {
            new SafeTensorBuilder()
                .AddTensor(input["keep"])
                .AddTensor(input["keep"], name: "keep.copy")
                .Save(target);
        }

        using SafeTensorFile output = SafeTensorFile.Open(target);

        Assert.Equal(2, output.Count);
        Assert.Equal(new float[] { 1, 2, 3, 4 }, output["keep"].ToArray<float>());
        Assert.Equal(new float[] { 1, 2, 3, 4 }, output["keep.copy"].ToArray<float>());
        Assert.Equal([2, 2], output["keep"].Shape);
    }

    [Fact]
    public void Writes_a_file_the_reader_accepts_with_the_strictest_options()
    {
        byte[] file = new SafeTensorBuilder()
            .AddTensor("a", new float[] { 1, 2 }, [2])
            .AddTensor("b", new byte[] { 7 }, [1])
            .AddTensor("c", new double[] { 3 }, [1])
            .ToByteArray();

        using SafeTensorFile model = SafeTensorFile.Read(
            file,
            new SafeTensorReadOptions { AllowNonContiguousData = false, AllowTrailingBytes = false });

        Assert.Equal(3, model.Count);
    }
}
