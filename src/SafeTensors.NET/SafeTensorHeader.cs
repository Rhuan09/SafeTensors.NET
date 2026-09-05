using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Text.Json;
using SafeTensors.Internal;

namespace SafeTensors
{
    /// <summary>
    /// The decoded and validated header of a SafeTensors file.
    /// </summary>
    /// <remarks>
    /// A SafeTensors file is a little-endian <c>uint64</c> header length, that many bytes of
    /// UTF-8 JSON, then the raw tensor bytes back to back. Everything a consumer needs to
    /// locate a tensor lives in the header, which is why reading it alone is enough to
    /// inspect a checkpoint of any size.
    /// </remarks>
    public sealed class SafeTensorHeader
    {
        /// <summary>
        /// The header size cap applied when no options are supplied. 100 MiB.
        /// </summary>
        public const long DefaultMaxHeaderSize = 100L * 1024 * 1024;

        /// <summary>Gets the length of the JSON header in bytes, excluding the 8-byte prefix.</summary>
        public long HeaderSize { get; }

        /// <summary>
        /// Gets the file offset at which tensor data begins: <see cref="HeaderSize"/> plus
        /// the 8-byte length prefix.
        /// </summary>
        public long DataOffset => HeaderSize + 8;

        /// <summary>
        /// Gets the byte length of the data section implied by the tensors, that is the end
        /// offset of the last tensor. Zero when the file holds no tensors.
        /// </summary>
        public long DataLength { get; }

        /// <summary>Gets the tensors, keyed by name.</summary>
        public IReadOnlyDictionary<string, TensorMetadata> Tensors { get; }

        /// <summary>
        /// Gets the string entries of <c>__metadata__</c>. Non-string values are kept as
        /// their raw JSON text.
        /// </summary>
        public IReadOnlyDictionary<string, string> Metadata { get; }

        /// <summary>
        /// Gets the raw <c>__metadata__</c> element, for callers that want to deserialise it
        /// into their own shape.
        /// </summary>
        public JsonElement? RawMetadata { get; }

        /// <summary>Creates a header from already-validated parts.</summary>
        public SafeTensorHeader(
            long headerSize,
            Dictionary<string, TensorMetadata> tensors,
            Dictionary<string, string>? metadata = null,
            JsonElement? rawMetadata = null)
        {
            if (tensors is null)
            {
                throw new ArgumentNullException(nameof(tensors));
            }

            HeaderSize = headerSize;
            Tensors = new ReadOnlyDictionary<string, TensorMetadata>(tensors);
            Metadata = new ReadOnlyDictionary<string, string>(metadata ?? new Dictionary<string, string>());
            RawMetadata = rawMetadata;

            long end = 0;
            foreach (KeyValuePair<string, TensorMetadata> pair in tensors)
            {
                if (pair.Value.DataEnd > end)
                {
                    end = pair.Value.DataEnd;
                }
            }

            DataLength = end;
        }

        /// <summary>Gets a metadata value, or <c>null</c> if the key is absent.</summary>
        public string? GetMetadata(string key)
            => Metadata.TryGetValue(key, out string? value) ? value : null;

        /// <summary>Tries to get a metadata value.</summary>
        public bool TryGetMetadata(string key, out string? value) => Metadata.TryGetValue(key, out value);

        /// <summary>
        /// Deserialises <c>__metadata__</c> into <typeparamref name="T"/>.
        /// </summary>
#if NET8_0_OR_GREATER
        [RequiresUnreferencedCode("Uses reflection-based JSON deserialisation. Pass a JsonSerializerOptions with a source-generated context, or read RawMetadata directly, in a trimmed app.")]
        [RequiresDynamicCode("Uses reflection-based JSON deserialisation, which needs runtime code generation.")]
#endif
        public T? DeserializeMetadata<T>(JsonSerializerOptions? options = null)
        {
            if (RawMetadata is null
                || RawMetadata.Value.ValueKind == JsonValueKind.Undefined
                || RawMetadata.Value.ValueKind == JsonValueKind.Null)
            {
                return default;
            }

            return JsonSerializer.Deserialize<T>(RawMetadata.Value.GetRawText(), options);
        }

        /// <summary>
        /// Reads and range-checks the 8-byte header length prefix at the current position.
        /// </summary>
        public static long ReadHeaderLength(Stream stream, SafeTensorReadOptions? options = null)
        {
            if (stream is null)
            {
                throw new ArgumentNullException(nameof(stream));
            }

            options ??= SafeTensorReadOptions.Default;

            Span<byte> prefix = stackalloc byte[8];
            BinaryUtils.ReadExactly(stream, prefix, "the 8-byte header length");
            return ValidateHeaderLength(BinaryUtils.ReadUInt64LittleEndian(prefix), options);
        }

        internal static long ValidateHeaderLength(ulong length, SafeTensorReadOptions options)
        {
            if (length == 0)
            {
                throw new SafeTensorCorruptHeaderException("Header length is zero; the file declares no JSON header.");
            }

            if (length > (ulong)options.MaxHeaderSize)
            {
                throw new SafeTensorCorruptHeaderException(
                    $"Header length ({length} bytes) exceeds the {options.MaxHeaderSize}-byte limit. " +
                    "Raise SafeTensorReadOptions.MaxHeaderSize if the file is genuinely this large.");
            }

            return (long)length;
        }

        /// <summary>
        /// Parses and validates a header.
        /// </summary>
        /// <param name="headerJson">UTF-8 JSON, without the length prefix.</param>
        /// <param name="headerSize">The declared header length, which must equal the JSON length.</param>
        /// <param name="totalSize">Total size of the file or buffer, or -1 when unknown.</param>
        /// <param name="options">Strictness settings; defaults are strict.</param>
        /// <exception cref="SafeTensorCorruptHeaderException">The JSON is malformed or a field has the wrong type.</exception>
        /// <exception cref="SafeTensorValidationException">The tensor layout is not internally consistent.</exception>
        public static SafeTensorHeader Parse(
            ReadOnlyMemory<byte> headerJson,
            long headerSize,
            long totalSize = -1,
            SafeTensorReadOptions? options = null)
        {
            options ??= SafeTensorReadOptions.Default;

            if (headerSize <= 0)
            {
                throw new SafeTensorCorruptHeaderException("Header length must be greater than zero.");
            }

            if (totalSize >= 0 && headerSize + 8 > totalSize)
            {
                throw new SafeTensorCorruptHeaderException(
                    $"Header length ({headerSize} + 8 bytes) exceeds the total size ({totalSize} bytes).");
            }

            var tensors = new Dictionary<string, TensorMetadata>(StringComparer.Ordinal);
            var metadata = new Dictionary<string, string>(StringComparer.Ordinal);
            JsonElement? rawMetadata = null;

            try
            {
                using JsonDocument document = JsonDocument.Parse(headerJson);
                JsonElement root = document.RootElement;

                if (root.ValueKind != JsonValueKind.Object)
                {
                    throw new SafeTensorCorruptHeaderException(
                        $"The header must be a JSON object, but it is {root.ValueKind}.");
                }

                foreach (JsonProperty property in root.EnumerateObject())
                {
                    if (property.NameEquals("__metadata__"))
                    {
                        rawMetadata = property.Value.Clone();
                        ReadMetadata(property.Value, metadata);
                        continue;
                    }

                    // JsonDocument keeps duplicate keys rather than collapsing them, so a
                    // crafted header can define one name twice with different offsets and
                    // have two consumers disagree about which one is real.
                    if (tensors.ContainsKey(property.Name))
                    {
                        throw new SafeTensorCorruptHeaderException(
                            $"Tensor '{property.Name}' is defined more than once in the header.");
                    }

                    tensors.Add(property.Name, ParseTensor(property.Name, property.Value));
                }
            }
            catch (JsonException ex)
            {
                throw new SafeTensorCorruptHeaderException("The header is not valid UTF-8 JSON.", ex);
            }

            long availableData = totalSize >= 0 ? totalSize - (headerSize + 8) : -1;
            ValidateLayout(tensors, availableData, options);

            return new SafeTensorHeader(headerSize, tensors, metadata, rawMetadata);
        }

        private static void ReadMetadata(JsonElement element, Dictionary<string, string> into)
        {
            if (element.ValueKind != JsonValueKind.Object)
            {
                return;
            }

            foreach (JsonProperty entry in element.EnumerateObject())
            {
                into[entry.Name] = entry.Value.ValueKind == JsonValueKind.String
                    ? entry.Value.GetString()!
                    : entry.Value.GetRawText();
            }
        }

        private static TensorMetadata ParseTensor(string name, JsonElement element)
        {
            if (element.ValueKind != JsonValueKind.Object)
            {
                throw new SafeTensorCorruptHeaderException(
                    $"Tensor '{name}' must be a JSON object, but it is {element.ValueKind}.");
            }

            if (!element.TryGetProperty("dtype", out JsonElement dtypeElement)
                || dtypeElement.ValueKind != JsonValueKind.String)
            {
                throw new SafeTensorCorruptHeaderException($"Tensor '{name}' has no string 'dtype'.");
            }

            string dtypeText = dtypeElement.GetString()!;
            if (!DTypes.TryParse(dtypeText, out SafeTensorDType dtype))
            {
                throw new SafeTensorCorruptHeaderException(
                    $"Tensor '{name}' declares an unsupported dtype '{dtypeText}'.");
            }

            if (!element.TryGetProperty("shape", out JsonElement shapeElement)
                || shapeElement.ValueKind != JsonValueKind.Array)
            {
                throw new SafeTensorCorruptHeaderException($"Tensor '{name}' has no 'shape' array.");
            }

            long[] shape = new long[shapeElement.GetArrayLength()];
            int index = 0;
            foreach (JsonElement dimension in shapeElement.EnumerateArray())
            {
                shape[index++] = ReadNonNegativeInt64(dimension, name, $"shape[{index - 1}]");
            }

            if (!element.TryGetProperty("data_offsets", out JsonElement offsetsElement)
                || offsetsElement.ValueKind != JsonValueKind.Array)
            {
                throw new SafeTensorCorruptHeaderException($"Tensor '{name}' has no 'data_offsets' array.");
            }

            if (offsetsElement.GetArrayLength() != 2)
            {
                throw new SafeTensorCorruptHeaderException(
                    $"Tensor '{name}' has {offsetsElement.GetArrayLength()} data_offsets; exactly 2 are required.");
            }

            long start = ReadNonNegativeInt64(offsetsElement[0], name, "data_offsets[0]");
            long end = ReadNonNegativeInt64(offsetsElement[1], name, "data_offsets[1]");

            return new TensorMetadata(name, dtype, shape, start, end);
        }

        /// <summary>
        /// Reads a non-negative int64, converting every way JSON can lie about a number into
        /// a header exception rather than letting <see cref="InvalidOperationException"/>
        /// escape the parser.
        /// </summary>
        private static long ReadNonNegativeInt64(JsonElement element, string tensorName, string field)
        {
            if (element.ValueKind != JsonValueKind.Number)
            {
                throw new SafeTensorCorruptHeaderException(
                    $"Tensor '{tensorName}' has a non-numeric {field} ({element.ValueKind}).");
            }

            if (!element.TryGetInt64(out long value))
            {
                throw new SafeTensorCorruptHeaderException(
                    $"Tensor '{tensorName}' has a {field} that is not a 64-bit integer: {element.GetRawText()}.");
            }

            if (value < 0)
            {
                throw new SafeTensorCorruptHeaderException(
                    $"Tensor '{tensorName}' has a negative {field}: {value}.");
            }

            return value;
        }

        /// <summary>
        /// Checks that the tensor byte ranges tile the data section: sorted by start offset
        /// they must begin at zero, never overlap, and stay inside the file.
        /// </summary>
        private static void ValidateLayout(
            Dictionary<string, TensorMetadata> tensors,
            long availableData,
            SafeTensorReadOptions options)
        {
            if (tensors.Count == 0)
            {
                return;
            }

            var ordered = new List<TensorMetadata>(tensors.Values);
            ordered.Sort(static (a, b) =>
            {
                int byStart = a.DataStart.CompareTo(b.DataStart);
                return byStart != 0 ? byStart : string.CompareOrdinal(a.Name, b.Name);
            });

            long cursor = 0;
            for (int i = 0; i < ordered.Count; i++)
            {
                TensorMetadata tensor = ordered[i];

                if (tensor.DataStart < cursor)
                {
                    // Two tensors sharing bytes is the format's sharpest edge: it lets one
                    // file hand the same memory to two names, so it is always an error.
                    throw new SafeTensorValidationException(
                        $"Tensor '{tensor.Name}' starts at {tensor.DataStart}, inside the range " +
                        $"already claimed by '{ordered[i - 1].Name}' which ends at {cursor}. " +
                        "Overlapping tensors are never valid.");
                }

                if (tensor.DataStart > cursor && !options.AllowNonContiguousData)
                {
                    throw new SafeTensorValidationException(
                        $"Tensor '{tensor.Name}' starts at {tensor.DataStart}, leaving " +
                        $"{tensor.DataStart - cursor} unclaimed bytes after offset {cursor}. " +
                        "Set SafeTensorReadOptions.AllowNonContiguousData to accept padded files.");
                }

                cursor = tensor.DataEnd;
            }

            if (availableData < 0)
            {
                return;
            }

            if (cursor > availableData)
            {
                throw new SafeTensorValidationException(
                    $"Tensor data needs {cursor} bytes but only {availableData} bytes follow the header.");
            }

            if (cursor < availableData && !options.AllowTrailingBytes)
            {
                throw new SafeTensorValidationException(
                    $"{availableData - cursor} bytes follow the last tensor. " +
                    "Set SafeTensorReadOptions.AllowTrailingBytes to accept them.");
            }
        }
    }
}
