using SafeTensors.Tests.Infrastructure;

namespace SafeTensors.Tests;

/// <summary>
/// What the reader does with headers that are malformed, hostile, or merely wrong.
/// </summary>
public class HeaderValidationTests
{
    [Fact]
    public void Rejects_overlapping_tensor_ranges()
    {
        // Two names claiming the same bytes. Accepting this lets one file hand the same
        // memory to two consumers that each believe they own it.
        byte[] file = Fixture.FromJson(
            """{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]},"b":{"dtype":"U8","shape":[8],"data_offsets":[4,12]}}""",
            dataBytes: 16);

        SafeTensorValidationException error =
            Assert.Throws<SafeTensorValidationException>(() => SafeTensorFile.Read(file));

        Assert.Contains("Overlapping", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Rejects_gaps_between_tensors_by_default()
    {
        byte[] file = Fixture.FromJson(
            """{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]},"b":{"dtype":"U8","shape":[8],"data_offsets":[40,48]}}""",
            dataBytes: 48);

        Assert.Throws<SafeTensorValidationException>(() => SafeTensorFile.Read(file));
    }

    [Fact]
    public void Accepts_gaps_when_the_caller_opts_in()
    {
        byte[] file = Fixture.FromJson(
            """{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]},"b":{"dtype":"U8","shape":[8],"data_offsets":[40,48]}}""",
            dataBytes: 48);

        using SafeTensorFile model = SafeTensorFile.Read(
            file,
            new SafeTensorReadOptions { AllowNonContiguousData = true });

        Assert.Equal(2, model.Count);
    }

    [Fact]
    public void Rejects_a_gap_before_the_first_tensor()
    {
        byte[] file = Fixture.FromJson(
            """{"a":{"dtype":"U8","shape":[8],"data_offsets":[8,16]}}""",
            dataBytes: 16);

        Assert.Throws<SafeTensorValidationException>(() => SafeTensorFile.Read(file));
    }

    [Theory]
    [InlineData("""{"a":{"dtype":"U8","shape":[8],"data_offsets":["0",8]}}""")]
    [InlineData("""{"a":{"dtype":"U8","shape":[8],"data_offsets":[null,8]}}""")]
    [InlineData("""{"a":{"dtype":"U8","shape":[8],"data_offsets":[{},8]}}""")]
    [InlineData("""{"a":{"dtype":"U8","shape":[8],"data_offsets":[0.5,8]}}""")]
    public void Reports_non_integer_offsets_as_a_header_error(string headerJson)
    {
        // The failure mode being guarded against is a raw InvalidOperationException from
        // JsonElement escaping the parser, which no caller would think to catch.
        byte[] file = Fixture.FromJson(headerJson, dataBytes: 16);

        Assert.Throws<SafeTensorCorruptHeaderException>(() => SafeTensorFile.Read(file));
    }

    [Theory]
    [InlineData("""{"a":{"dtype":"U8","shape":[8],"data_offsets":[0]}}""")]
    [InlineData("""{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,4,8]}}""")]
    [InlineData("""{"a":{"dtype":"U8","shape":[8],"data_offsets":[]}}""")]
    public void Requires_exactly_two_offsets(string headerJson)
    {
        byte[] file = Fixture.FromJson(headerJson, dataBytes: 16);

        Assert.Throws<SafeTensorCorruptHeaderException>(() => SafeTensorFile.Read(file));
    }

    [Fact]
    public void Rejects_a_shape_that_disagrees_with_the_byte_range()
    {
        byte[] file = Fixture.FromJson(
            """{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,8]}}""",
            dataBytes: 8);

        Assert.Throws<SafeTensorValidationException>(() => SafeTensorFile.Read(file));
    }

    [Fact]
    public void Rejects_a_shape_whose_element_count_overflows()
    {
        // 2^62 x 2^62 wraps to a small positive number in unchecked arithmetic, which would
        // then agree with a tiny byte range and pass validation.
        byte[] file = Fixture.FromJson(
            """{"a":{"dtype":"U8","shape":[4611686018427387904,4611686018427387904],"data_offsets":[0,8]}}""",
            dataBytes: 8);

        Assert.Throws<SafeTensorValidationException>(() => SafeTensorFile.Read(file));
    }

    [Fact]
    public void Rejects_negative_dimensions()
    {
        byte[] file = Fixture.FromJson(
            """{"a":{"dtype":"U8","shape":[-1,8],"data_offsets":[0,8]}}""",
            dataBytes: 8);

        Assert.ThrowsAny<SafeTensorException>(() => SafeTensorFile.Read(file));
    }

    [Fact]
    public void Rejects_data_that_runs_past_the_end_of_the_file()
    {
        byte[] file = Fixture.FromJson(
            """{"a":{"dtype":"U8","shape":[64],"data_offsets":[0,64]}}""",
            dataBytes: 8);

        Assert.Throws<SafeTensorValidationException>(() => SafeTensorFile.Read(file));
    }

    [Fact]
    public void Rejects_an_unknown_dtype()
    {
        byte[] file = Fixture.FromJson(
            """{"a":{"dtype":"F4","shape":[8],"data_offsets":[0,4]}}""",
            dataBytes: 8);

        Assert.Throws<SafeTensorCorruptHeaderException>(() => SafeTensorFile.Read(file));
    }

    [Fact]
    public void Rejects_a_header_larger_than_the_configured_limit()
    {
        byte[] file = Fixture.WithDeclaredHeaderLength("{}", declaredLength: 1UL << 40, dataBytes: 8);

        SafeTensorCorruptHeaderException error =
            Assert.Throws<SafeTensorCorruptHeaderException>(() => SafeTensorFile.Read(file));

        Assert.Contains("MaxHeaderSize", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Rejects_a_zero_length_header()
    {
        byte[] file = Fixture.WithDeclaredHeaderLength("", declaredLength: 0, dataBytes: 8);

        Assert.Throws<SafeTensorCorruptHeaderException>(() => SafeTensorFile.Read(file));
    }

    [Fact]
    public void Rejects_a_file_shorter_than_the_length_prefix()
    {
        Assert.Throws<SafeTensorCorruptHeaderException>(() => SafeTensorFile.Read(new byte[4]));
    }

    [Fact]
    public void Rejects_a_json_root_that_is_not_an_object()
    {
        byte[] file = Fixture.FromJson("[1,2,3]", dataBytes: 8);

        Assert.Throws<SafeTensorCorruptHeaderException>(() => SafeTensorFile.Read(file));
    }

    [Fact]
    public void Rejects_malformed_json()
    {
        byte[] file = Fixture.FromJson("{not json", dataBytes: 8);

        Assert.Throws<SafeTensorCorruptHeaderException>(() => SafeTensorFile.Read(file));
    }

    [Fact]
    public void Accepts_a_scalar_with_an_empty_shape()
    {
        byte[] file = Fixture.FromJson(
            """{"a":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}""",
            dataBytes: 4);

        using SafeTensorFile model = SafeTensorFile.Read(file);

        Assert.Equal(1, model["a"].ElementCount);
        Assert.Equal(0, model["a"].Rank);
    }

    [Fact]
    public void Accepts_a_zero_element_tensor()
    {
        byte[] file = Fixture.FromJson(
            """{"a":{"dtype":"F32","shape":[0,4],"data_offsets":[0,0]}}""",
            dataBytes: 0);

        using SafeTensorFile model = SafeTensorFile.Read(file);

        Assert.Equal(0, model["a"].ElementCount);
        Assert.Equal(0, model["a"].ByteLength);
    }

    [Fact]
    public void Accepts_trailing_bytes_by_default_and_rejects_them_on_request()
    {
        byte[] file = Fixture.FromJson(
            """{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}""",
            dataBytes: 32);

        using (SafeTensorFile lenient = SafeTensorFile.Read(file))
        {
            Assert.Single(lenient.Tensors);
        }

        Assert.Throws<SafeTensorValidationException>(
            () => SafeTensorFile.Read(file, new SafeTensorReadOptions { AllowTrailingBytes = false }));
    }

    [Fact]
    public void Reads_metadata_and_leaves_it_out_of_the_tensor_list()
    {
        byte[] file = Fixture.FromJson(
            """{"__metadata__":{"format":"pt","step":"1200"},"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}""",
            dataBytes: 8);

        using SafeTensorFile model = SafeTensorFile.Read(file);

        Assert.Single(model.Tensors);
        Assert.Equal("pt", model.Metadata["format"]);
        Assert.Equal("1200", model.Metadata["step"]);
    }

    [Fact]
    public void Rejects_a_file_with_duplicate_tensor_keys()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestFiles", "duplicate_keys_in_header.safetensors");

        Assert.Throws<SafeTensorCorruptHeaderException>(() => SafeTensorFile.Open(path));
    }

    [Fact]
    public void Rejects_an_empty_file()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestFiles", "zero_len_file.safetensors");

        Assert.Throws<SafeTensorCorruptHeaderException>(() => SafeTensorFile.Open(path));
    }

    [Fact]
    public void Rejects_a_header_length_that_exceeds_the_file()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestFiles", "header_size_too_big.safetensors");

        Assert.Throws<SafeTensorCorruptHeaderException>(() => SafeTensorFile.Open(path));
    }
}
