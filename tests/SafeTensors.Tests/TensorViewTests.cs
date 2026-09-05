namespace SafeTensors.Tests;

/// <summary>
/// Accessors on a single tensor: spans, windows, slices and conversions.
/// </summary>
public class TensorViewTests
{
    private static SafeTensorFile Matrix()
    {
        // 4 rows of 3 columns, values 0..11, so a row's contents identify it.
        float[] values = Enumerable.Range(0, 12).Select(i => (float)i).ToArray();

        return SafeTensorFile.Read(
            new SafeTensorBuilder().AddTensor("m", values, [4, 3]).ToByteArray());
    }

    [Fact]
    public void Slices_the_outermost_dimension_without_copying()
    {
        using SafeTensorFile model = Matrix();

        TensorView rows = model["m"].Slice(1, 2);

        Assert.Equal([2, 3], rows.Shape);
        Assert.Equal(new float[] { 3, 4, 5, 6, 7, 8 }, rows.ToArray<float>());
        Assert.True(rows.IsZeroCopy);
    }

    [Fact]
    public void Slicing_a_slice_stays_consistent()
    {
        using SafeTensorFile model = Matrix();

        TensorView row = model["m"].Slice(1, 3).Slice(2, 1);

        Assert.Equal(new float[] { 9, 10, 11 }, row.ToArray<float>());
    }

    [Theory]
    [InlineData(-1, 1)]
    [InlineData(0, 5)]
    [InlineData(4, 1)]
    [InlineData(2, -1)]
    public void Rejects_slices_outside_the_first_dimension(long start, long count)
    {
        using SafeTensorFile model = Matrix();

        Assert.Throws<ArgumentOutOfRangeException>(() => model["m"].Slice(start, count));
    }

    [Fact]
    public void Refuses_to_slice_a_scalar()
    {
        using SafeTensorFile model = SafeTensorFile.Read(
            new SafeTensorBuilder().AddTensor("s", new float[] { 42 }, []).ToByteArray());

        Assert.Throws<InvalidOperationException>(() => model["s"].Slice(0, 1));
    }

    [Fact]
    public void Reads_a_window_of_elements()
    {
        using SafeTensorFile model = Matrix();

        Assert.Equal(new float[] { 4, 5, 6 }, model["m"].AsSpan<float>(4, 3).ToArray());
    }

    [Theory]
    [InlineData(0, 13)]
    [InlineData(12, 1)]
    [InlineData(10, 5)]
    public void Rejects_windows_that_run_past_the_end(long offset, int count)
    {
        using SafeTensorFile model = Matrix();

        Assert.Throws<ArgumentOutOfRangeException>(() => model["m"].AsSpan<float>(offset, count).Length);
    }

    [Fact]
    public void Reinterprets_bytes_as_another_element_type()
    {
        using SafeTensorFile model = SafeTensorFile.Read(
            new SafeTensorBuilder().AddTensor("f", new[] { 1.0f }, [1]).ToByteArray());

        Assert.Equal(0x3F800000u, model["f"].AsSpan<uint>()[0]);
    }

    [Fact]
    public void Converts_every_numeric_dtype_to_float()
    {
        byte[] file = new SafeTensorBuilder()
            .AddTensor("f32", new[] { 1.5f, -2.5f }, [2])
            .AddTensor("f64", new[] { 1.5d, -2.5d }, [2])
            .AddTensor("f16", new[] { (Float16)1.5f, (Float16)(-2.5f) }, [2])
            .AddTensor("bf16", new[] { (BFloat16)1.5f, (BFloat16)(-2.5f) }, [2])
            .AddTensor("i32", new[] { 1, -2 }, [2])
            .AddTensor("u8", new byte[] { 1, 2 }, [2])
            .AddTensor("i64", new[] { 1L, -2L }, [2])
            .AddTensor("bool", new[] { true, false }, [2])
            .ToByteArray();

        using SafeTensorFile model = SafeTensorFile.Read(file);

        Assert.Equal([1.5f, -2.5f], model["f32"].ToSingleArray());
        Assert.Equal([1.5f, -2.5f], model["f64"].ToSingleArray());
        Assert.Equal([1.5f, -2.5f], model["f16"].ToSingleArray());
        Assert.Equal([1.5f, -2.5f], model["bf16"].ToSingleArray());
        Assert.Equal([1f, -2f], model["i32"].ToSingleArray());
        Assert.Equal([1f, 2f], model["u8"].ToSingleArray());
        Assert.Equal([1f, -2f], model["i64"].ToSingleArray());
        Assert.Equal([1f, 0f], model["bool"].ToSingleArray());
    }

    [Fact]
    public void Refuses_to_convert_a_dtype_with_no_float_meaning()
    {
        byte[] file = new SafeTensorBuilder()
            .AddTensor("fp8", SafeTensorDType.F8_E4M3, new byte[] { 0x38, 0x40 }, [2])
            .ToByteArray();

        using SafeTensorFile model = SafeTensorFile.Read(file);

        Assert.Throws<NotSupportedException>(() => model["fp8"].ToSingleArray());
    }

    [Fact]
    public void Streams_a_tensor()
    {
        using SafeTensorFile model = Matrix();
        using Stream stream = model["m"].OpenStream();

        Assert.Equal(48, stream.Length);

        var buffer = new byte[48];
        int read = 0;
        while (read < buffer.Length)
        {
            int got = stream.Read(buffer, read, buffer.Length - read);
            Assert.NotEqual(0, got);
            read += got;
        }

        Assert.Equal(11f, BitConverter.ToSingle(buffer, 44));
    }

    [Fact]
    public void Copies_into_a_caller_owned_buffer()
    {
        using SafeTensorFile model = Matrix();

        var destination = new float[12];
        model["m"].CopyTo(destination.AsSpan());

        Assert.Equal(11f, destination[11]);
    }

    [Fact]
    public void Rejects_a_destination_that_is_too_small()
    {
        using SafeTensorFile model = Matrix();

        Assert.Throws<ArgumentException>(() => model["m"].CopyTo(new float[3].AsSpan()));
    }

    [Fact]
    public void A_memory_backed_file_has_no_stable_address()
    {
        using SafeTensorFile model = Matrix();

        unsafe
        {
            // Managed buffers move, so the view declines rather than promising an address
            // it cannot keep. Only the memory-mapped path returns a pointer.
            Assert.True(model["m"].DangerousGetPointer() is null);
        }
    }

    [Fact]
    public void Reports_whether_a_tensor_fits_in_one_span()
    {
        using SafeTensorFile model = Matrix();

        Assert.True(model["m"].FitsInSingleSpan);
    }
}
