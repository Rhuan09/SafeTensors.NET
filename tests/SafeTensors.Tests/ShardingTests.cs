using System.Text.Json;
using SafeTensors.Tests.Infrastructure;

namespace SafeTensors.Tests;

/// <summary>
/// Multi-file models, and what the loader does with an index it should not trust.
/// </summary>
public class ShardingTests
{
    private static string WriteModel(TempDirectory directory)
    {
        new SafeTensorBuilder()
            .AddTensor("layer.0.weight", new float[] { 1, 2, 3, 4 }, [2, 2])
            .Save(directory.File("model-00001-of-00002.safetensors"));

        new SafeTensorBuilder()
            .AddTensor("layer.1.weight", new float[] { 5, 6 }, [2])
            .Save(directory.File("model-00002-of-00002.safetensors"));

        string index = directory.File("model.safetensors.index.json");
        File.WriteAllText(index, """
            {
              "metadata": { "total_size": "24" },
              "weight_map": {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors"
              }
            }
            """);

        return index;
    }

    [Fact]
    public void Reads_tensors_from_the_shard_that_holds_them()
    {
        using TempDirectory directory = Infrastructure.Fixture.NewDirectory();
        using ShardedSafeTensorFile model = ShardedSafeTensorFile.Open(WriteModel(directory));

        Assert.Equal(2, model.Count);
        Assert.Equal(new float[] { 1, 2, 3, 4 }, model["layer.0.weight"].ToArray<float>());
        Assert.Equal(new float[] { 5, 6 }, model["layer.1.weight"].ToArray<float>());
        Assert.Equal("24", model.Index.Metadata["total_size"]);
    }

    [Fact]
    public void Reports_a_name_the_index_does_not_list()
    {
        using TempDirectory directory = Infrastructure.Fixture.NewDirectory();
        using ShardedSafeTensorFile model = ShardedSafeTensorFile.Open(WriteModel(directory));

        Assert.Throws<SafeTensorNotFoundException>(() => model["absent"]);
        Assert.False(model.TryGetTensor("absent", out _));
    }

    [Fact]
    public void A_missing_shard_is_an_error_not_a_missing_tensor()
    {
        using TempDirectory directory = Infrastructure.Fixture.NewDirectory();
        string index = WriteModel(directory);
        File.Delete(directory.File("model-00002-of-00002.safetensors"));

        using ShardedSafeTensorFile model = ShardedSafeTensorFile.Open(index);

        // Reporting a broken download as "tensor not found" sends whoever is debugging it
        // looking in exactly the wrong place.
        Assert.Throws<FileNotFoundException>(() => model.TryGetTensor("layer.1.weight", out _));
    }

    [Theory]
    [InlineData("../escape.safetensors")]
    [InlineData("../../../../etc/shadow")]
    [InlineData("sub/../../escape.safetensors")]
    public void Refuses_a_shard_name_that_escapes_the_model_directory(string shard)
    {
        using TempDirectory directory = Infrastructure.Fixture.NewDirectory();

        string index = directory.File("model.safetensors.index.json");
        File.WriteAllText(index, JsonSerializer.Serialize(new
        {
            weight_map = new Dictionary<string, string> { ["w"] = shard }
        }));

        using ShardedSafeTensorFile model = ShardedSafeTensorFile.Open(index);

        // The index arrives with the download. A weight_map entry is a file name chosen by
        // whoever published the model, so it gets treated as untrusted input.
        SafeTensorValidationException error =
            Assert.Throws<SafeTensorValidationException>(() => model.GetTensor("w"));

        Assert.Contains("outside the model directory", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Refuses_an_absolute_shard_path()
    {
        using TempDirectory directory = Infrastructure.Fixture.NewDirectory();

        string absolute = Path.Combine(Path.GetTempPath(), "elsewhere.safetensors");
        string index = directory.File("model.safetensors.index.json");
        File.WriteAllText(index, JsonSerializer.Serialize(new
        {
            weight_map = new Dictionary<string, string> { ["w"] = absolute }
        }));

        using ShardedSafeTensorFile model = ShardedSafeTensorFile.Open(index);

        SafeTensorValidationException error =
            Assert.Throws<SafeTensorValidationException>(() => model.GetTensor("w"));

        Assert.Contains("absolute path", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Allows_a_shard_in_a_subdirectory_of_the_model()
    {
        using TempDirectory directory = Infrastructure.Fixture.NewDirectory();

        Directory.CreateDirectory(Path.Combine(directory.Path, "weights"));
        new SafeTensorBuilder()
            .AddTensor("w", new float[] { 1 }, [1])
            .Save(Path.Combine(directory.Path, "weights", "shard.safetensors"));

        string index = directory.File("model.safetensors.index.json");
        File.WriteAllText(index, JsonSerializer.Serialize(new
        {
            weight_map = new Dictionary<string, string> { ["w"] = "weights/shard.safetensors" }
        }));

        using ShardedSafeTensorFile model = ShardedSafeTensorFile.Open(index);

        Assert.Equal(new float[] { 1 }, model["w"].ToArray<float>());
    }

    [Fact]
    public void Opens_each_shard_once_under_concurrent_access()
    {
        using TempDirectory directory = Infrastructure.Fixture.NewDirectory();
        using ShardedSafeTensorFile model = ShardedSafeTensorFile.Open(WriteModel(directory));

        // A bare ConcurrentDictionary factory can run on several threads and keep only one
        // result, silently leaking a memory mapping for every loser. Same instance every
        // time is the observable proof that does not happen.
        SafeTensorFile[] opened = new SafeTensorFile[32];
        Parallel.For(0, opened.Length, i =>
        {
            opened[i] = model.OpenShard("model-00001-of-00002.safetensors");
        });

        Assert.All(opened, shard => Assert.Same(opened[0], shard));
    }

    [Fact]
    public void Reads_the_same_tensor_from_many_threads()
    {
        using TempDirectory directory = Infrastructure.Fixture.NewDirectory();
        using ShardedSafeTensorFile model = ShardedSafeTensorFile.Open(WriteModel(directory));

        Parallel.For(0, 64, _ =>
        {
            Assert.Equal(new float[] { 1, 2, 3, 4 }, model["layer.0.weight"].ToArray<float>());
        });
    }

    [Fact]
    public void Rejects_an_index_without_a_weight_map()
    {
        Assert.Throws<SafeTensorCorruptHeaderException>(() => ShardIndex.Parse("""{"metadata":{}}"""));
    }

    [Fact]
    public void Rejects_a_weight_map_entry_that_is_not_a_file_name()
    {
        Assert.Throws<SafeTensorCorruptHeaderException>(
            () => ShardIndex.Parse("""{"weight_map":{"w":["a.safetensors"]}}"""));
    }

    [Fact]
    public void Lists_each_shard_file_once()
    {
        ShardIndex index = ShardIndex.Parse("""
            {"weight_map":{"a":"s1.safetensors","b":"s1.safetensors","c":"s2.safetensors"}}
            """);

        Assert.Equal(["s1.safetensors", "s2.safetensors"], index.ShardFiles);
    }

    [Fact]
    public void Using_a_disposed_model_throws()
    {
        using TempDirectory directory = Infrastructure.Fixture.NewDirectory();
        ShardedSafeTensorFile model = ShardedSafeTensorFile.Open(WriteModel(directory));

        model.Dispose();

        Assert.Throws<ObjectDisposedException>(() => model["layer.0.weight"]);
    }
}
