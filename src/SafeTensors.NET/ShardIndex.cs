using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Text.Json;

namespace SafeTensors
{
    /// <summary>
    /// A <c>model.safetensors.index.json</c>: the map from tensor name to the shard file
    /// that holds it.
    /// </summary>
    /// <remarks>
    /// Models past a few gigabytes are published as numbered shards with one small index
    /// naming which file each tensor lives in. The index is the only part that has to be
    /// read up front.
    /// </remarks>
    public sealed class ShardIndex
    {
        /// <summary>Gets the map from tensor name to shard file name.</summary>
        public IReadOnlyDictionary<string, string> WeightMap { get; }

        /// <summary>Gets the index file's own metadata, such as <c>total_size</c>.</summary>
        public IReadOnlyDictionary<string, string> Metadata { get; }

        /// <summary>Gets the distinct shard file names, in the order first seen.</summary>
        public IReadOnlyList<string> ShardFiles { get; }

        /// <summary>Creates an index from an already-built weight map.</summary>
        public ShardIndex(Dictionary<string, string> weightMap, Dictionary<string, string>? metadata = null)
        {
            if (weightMap is null)
            {
                throw new ArgumentNullException(nameof(weightMap));
            }

            WeightMap = new ReadOnlyDictionary<string, string>(weightMap);
            Metadata = new ReadOnlyDictionary<string, string>(metadata ?? new Dictionary<string, string>());

            var seen = new HashSet<string>(StringComparer.Ordinal);
            var order = new List<string>();
            foreach (string shard in weightMap.Values)
            {
                if (seen.Add(shard))
                {
                    order.Add(shard);
                }
            }

            ShardFiles = new ReadOnlyCollection<string>(order);
        }

        /// <summary>Loads an index from a JSON file.</summary>
        public static ShardIndex Load(string path)
        {
            if (string.IsNullOrEmpty(path))
            {
                throw new ArgumentException("Path cannot be null or empty.", nameof(path));
            }

            if (!File.Exists(path))
            {
                throw new FileNotFoundException("Shard index file not found.", path);
            }

            return Parse(File.ReadAllText(path));
        }

        /// <summary>Parses an index from JSON text.</summary>
        public static ShardIndex Parse(string json)
        {
            if (string.IsNullOrEmpty(json))
            {
                throw new ArgumentException("Index JSON cannot be null or empty.", nameof(json));
            }

            var weightMap = new Dictionary<string, string>(StringComparer.Ordinal);
            var metadata = new Dictionary<string, string>(StringComparer.Ordinal);

            try
            {
                using JsonDocument document = JsonDocument.Parse(json);
                JsonElement root = document.RootElement;

                if (root.ValueKind != JsonValueKind.Object)
                {
                    throw new SafeTensorCorruptHeaderException(
                        $"A shard index must be a JSON object, but it is {root.ValueKind}.");
                }

                if (!root.TryGetProperty("weight_map", out JsonElement weightMapElement)
                    || weightMapElement.ValueKind != JsonValueKind.Object)
                {
                    throw new SafeTensorCorruptHeaderException("A shard index must contain a 'weight_map' object.");
                }

                foreach (JsonProperty entry in weightMapElement.EnumerateObject())
                {
                    if (entry.Value.ValueKind != JsonValueKind.String)
                    {
                        throw new SafeTensorCorruptHeaderException(
                            $"weight_map entry '{entry.Name}' must name a shard file as a string, " +
                            $"but it is {entry.Value.ValueKind}.");
                    }

                    weightMap[entry.Name] = entry.Value.GetString()!;
                }

                if (root.TryGetProperty("metadata", out JsonElement metadataElement)
                    && metadataElement.ValueKind == JsonValueKind.Object)
                {
                    foreach (JsonProperty entry in metadataElement.EnumerateObject())
                    {
                        metadata[entry.Name] = entry.Value.ValueKind == JsonValueKind.String
                            ? entry.Value.GetString()!
                            : entry.Value.GetRawText();
                    }
                }
            }
            catch (JsonException ex)
            {
                throw new SafeTensorCorruptHeaderException("The shard index is not valid JSON.", ex);
            }

            return new ShardIndex(weightMap, metadata);
        }
    }
}
