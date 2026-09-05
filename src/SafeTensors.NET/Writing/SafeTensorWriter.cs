using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using SafeTensors.Internal;

namespace SafeTensors
{
    /// <summary>
    /// Writes tensors and metadata in the SafeTensors layout.
    /// </summary>
    /// <remarks>
    /// The header is sized from the JSON it actually produces, so there is no fixed budget
    /// to overflow: a model with a hundred thousand tensors writes just as correctly as one
    /// with three.
    /// </remarks>
    public static class SafeTensorWriter
    {
        private const int PaddingByte = 0x20;

        /// <summary>
        /// Writes tensors to a stream.
        /// </summary>
        /// <param name="destination">Where to write. Written sequentially from the current position.</param>
        /// <param name="tensors">Tensors, written in the order given.</param>
        /// <param name="metadata">Optional <c>__metadata__</c> entries.</param>
        /// <param name="alignTo8Bytes">
        /// Pad the header with spaces so tensor data starts on an 8-byte boundary. On by
        /// default, matching what the reference implementations produce, and worth keeping:
        /// unaligned data defeats vectorised reads on the consuming side.
        /// </param>
        /// <exception cref="SafeTensorValidationException">Two tensors share a name.</exception>
        public static void Write(
            Stream destination,
            IEnumerable<TensorItem> tensors,
            IReadOnlyDictionary<string, string>? metadata = null,
            bool alignTo8Bytes = true)
        {
            if (destination is null)
            {
                throw new ArgumentNullException(nameof(destination));
            }

            if (tensors is null)
            {
                throw new ArgumentNullException(nameof(tensors));
            }

            var items = new List<TensorItem>(tensors);
            var offsets = new (long Start, long End)[items.Count];
            var seen = new HashSet<string>(StringComparer.Ordinal);

            long cursor = 0;
            for (int i = 0; i < items.Count; i++)
            {
                TensorItem item = items[i];

                // JSON objects have no duplicate-key rule that readers agree on, so writing
                // one name twice produces a file whose contents depend on the reader.
                if (!seen.Add(item.Name))
                {
                    throw new SafeTensorValidationException(
                        $"Tensor '{item.Name}' was added twice. Tensor names must be unique within a file.");
                }

                long end = checked(cursor + item.ByteLength);
                offsets[i] = (cursor, end);
                cursor = end;
            }

            byte[] header = BuildHeader(items, offsets, metadata);
            int padding = alignTo8Bytes ? (int)((8 - (header.Length % 8)) % 8) : 0;

            Span<byte> prefix = stackalloc byte[8];
            BinaryUtils.WriteUInt64LittleEndian(prefix, (ulong)(header.Length + padding));
            StreamWrite.Span(destination, prefix);

            destination.Write(header, 0, header.Length);

            if (padding > 0)
            {
                Span<byte> pad = stackalloc byte[8];
                pad.Fill(PaddingByte);
                StreamWrite.Span(destination, pad.Slice(0, padding));
            }

            for (int i = 0; i < items.Count; i++)
            {
                items[i].WriteTo(destination);
            }

            destination.Flush();
        }

        /// <summary>
        /// Writes tensors to a file, replacing any existing file only once the new one is
        /// complete and on disk.
        /// </summary>
        /// <remarks>
        /// The bytes go to a temporary file in the same directory, are flushed through to
        /// the device, and only then replace the target in a single rename. A crash at any
        /// point leaves either the old file or the temporary one — never a half-written
        /// checkpoint under the name something else is about to load.
        /// </remarks>
        public static void WriteFile(
            string path,
            IEnumerable<TensorItem> tensors,
            IReadOnlyDictionary<string, string>? metadata = null,
            bool alignTo8Bytes = true)
        {
            if (string.IsNullOrEmpty(path))
            {
                throw new ArgumentException("Path cannot be null or empty.", nameof(path));
            }

            string fullPath = Path.GetFullPath(path);
            string directory = Path.GetDirectoryName(fullPath)!;

            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            // Same directory, so the replace below is a rename within one volume.
            string temporary = Path.Combine(directory, $".{Path.GetFileName(fullPath)}.{Guid.NewGuid():N}.tmp");

            try
            {
                using (var file = new FileStream(
                    temporary,
                    FileMode.CreateNew,
                    FileAccess.Write,
                    FileShare.None,
                    bufferSize: 1 << 16,
                    FileOptions.SequentialScan))
                {
                    Write(file, tensors, metadata, alignTo8Bytes);

                    // Without this the rename can land before the data does, and a power
                    // loss leaves a correctly named file full of nothing.
                    file.Flush(flushToDisk: true);
                }

                Replace(temporary, fullPath);
            }
            catch
            {
                TryDelete(temporary);
                throw;
            }
        }

        /// <summary>Serialises tensors into a new byte array.</summary>
        public static byte[] WriteToBytes(
            IEnumerable<TensorItem> tensors,
            IReadOnlyDictionary<string, string>? metadata = null,
            bool alignTo8Bytes = true)
        {
            using var buffer = new MemoryStream();
            Write(buffer, tensors, metadata, alignTo8Bytes);
            return buffer.ToArray();
        }

        private static byte[] BuildHeader(
            List<TensorItem> items,
            (long Start, long End)[] offsets,
            IReadOnlyDictionary<string, string>? metadata)
        {
            using var buffer = new MemoryStream();

            using (var json = new Utf8JsonWriter(buffer))
            {
                json.WriteStartObject();

                if (metadata is not null && metadata.Count > 0)
                {
                    json.WriteStartObject(TensorItem.MetadataKey);
                    foreach (KeyValuePair<string, string> entry in metadata)
                    {
                        json.WriteString(entry.Key, entry.Value ?? string.Empty);
                    }

                    json.WriteEndObject();
                }

                for (int i = 0; i < items.Count; i++)
                {
                    TensorItem item = items[i];

                    json.WriteStartObject(item.Name);
                    json.WriteString("dtype", DTypes.ToHeaderString(item.DType));

                    json.WriteStartArray("shape");
                    for (int d = 0; d < item.Shape.Length; d++)
                    {
                        json.WriteNumberValue(item.Shape[d]);
                    }

                    json.WriteEndArray();

                    json.WriteStartArray("data_offsets");
                    json.WriteNumberValue(offsets[i].Start);
                    json.WriteNumberValue(offsets[i].End);
                    json.WriteEndArray();

                    json.WriteEndObject();
                }

                json.WriteEndObject();
            }

            return buffer.ToArray();
        }

        /// <summary>
        /// Moves <paramref name="source"/> onto <paramref name="destination"/> atomically
        /// where the platform allows it.
        /// </summary>
        private static void Replace(string source, string destination)
        {
#if NETSTANDARD2_0
            if (File.Exists(destination))
            {
                // File.Replace is a single atomic operation on both Windows and Unix.
                File.Replace(source, destination, destinationBackupFileName: null, ignoreMetadataErrors: true);
            }
            else
            {
                File.Move(source, destination);
            }
#else
            // MoveFileEx with REPLACE_EXISTING on Windows, rename(2) on Unix: the target is
            // never absent, unlike a Delete-then-Move.
            File.Move(source, destination, overwrite: true);
#endif
        }

        private static void TryDelete(string path)
        {
            try
            {
                if (File.Exists(path))
                {
                    File.Delete(path);
                }
            }
            catch (IOException)
            {
                // The original failure is what the caller needs to see.
            }
            catch (UnauthorizedAccessException)
            {
            }
        }
    }
}
