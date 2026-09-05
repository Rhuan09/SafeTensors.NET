using System;
using System.Collections.Generic;
using System.IO;

namespace SafeTensors
{
    /// <summary>
    /// Collects tensors and metadata, then writes them as a SafeTensors file.
    /// </summary>
    /// <remarks>
    /// Tensors are written in the order they are added. The builder holds references to the
    /// arrays you give it rather than copying them, so it costs no extra memory to stage a
    /// whole model — and so the arrays must not change until the file is written.
    /// </remarks>
    public sealed class SafeTensorBuilder
    {
        private readonly List<TensorItem> _tensors = new List<TensorItem>();
        private readonly Dictionary<string, string> _metadata = new Dictionary<string, string>(StringComparer.Ordinal);
        private bool _alignTo8Bytes = true;

        /// <summary>Gets the tensors added so far.</summary>
        public IReadOnlyList<TensorItem> Tensors => _tensors;

        /// <summary>Gets the number of tensors added so far.</summary>
        public int Count => _tensors.Count;

        /// <summary>Adds or replaces a <c>__metadata__</c> entry.</summary>
        public SafeTensorBuilder WithMetadata(string key, string value)
        {
            if (string.IsNullOrEmpty(key))
            {
                throw new ArgumentException("Metadata key cannot be null or empty.", nameof(key));
            }

            _metadata[key] = value ?? string.Empty;
            return this;
        }

        /// <summary>Adds or replaces several <c>__metadata__</c> entries.</summary>
        public SafeTensorBuilder WithMetadata(IEnumerable<KeyValuePair<string, string>> metadata)
        {
            if (metadata is null)
            {
                throw new ArgumentNullException(nameof(metadata));
            }

            foreach (KeyValuePair<string, string> entry in metadata)
            {
                WithMetadata(entry.Key, entry.Value);
            }

            return this;
        }

        /// <summary>
        /// Sets whether to pad the header so tensor data starts on an 8-byte boundary.
        /// On by default.
        /// </summary>
        public SafeTensorBuilder With8ByteAlignment(bool align)
        {
            _alignTo8Bytes = align;
            return this;
        }

        /// <summary>Adds a tensor from a typed array. The array is referenced, not copied.</summary>
        public SafeTensorBuilder AddTensor<T>(string name, T[] data, long[] shape)
            where T : unmanaged
        {
            _tensors.Add(TensorItem.FromArray(name, data, shape));
            return this;
        }

        /// <summary>Adds a tensor from typed memory. The memory is referenced, not copied.</summary>
        public SafeTensorBuilder AddTensor<T>(string name, ReadOnlyMemory<T> data, long[] shape)
            where T : unmanaged
        {
            _tensors.Add(TensorItem.FromMemory(name, data, shape));
            return this;
        }

        /// <summary>Adds a tensor from a typed span. This overload copies, because a span cannot be stored.</summary>
        public SafeTensorBuilder AddTensor<T>(string name, ReadOnlySpan<T> data, long[] shape)
            where T : unmanaged
        {
            _tensors.Add(TensorItem.FromSpan(name, data, shape));
            return this;
        }

        /// <summary>Adds a tensor from raw bytes with an explicit dtype.</summary>
        public SafeTensorBuilder AddTensor(string name, SafeTensorDType dtype, ReadOnlyMemory<byte> data, long[] shape)
        {
            _tensors.Add(TensorItem.FromBytes(name, dtype, shape, data));
            return this;
        }

        /// <summary>
        /// Adds a tensor copied from an open file, without materialising it in managed memory.
        /// </summary>
        /// <remarks>
        /// Use this to rewrite, merge or reshard checkpoints: the bytes stream from the
        /// source file to the destination when the file is written, so the cost does not
        /// scale with how many tensors you carry over.
        /// </remarks>
        public SafeTensorBuilder AddTensor(TensorView source, string? name = null)
        {
            if (source is null)
            {
                throw new ArgumentNullException(nameof(source));
            }

            _tensors.Add(new TensorItem(
                name ?? source.Name,
                source.DType,
                source.Metadata.ToShapeArray(),
                source.ByteLength,
                destination =>
                {
                    using Stream input = source.OpenStream();
                    input.CopyTo(destination);
                }));

            return this;
        }

        /// <summary>Adds a pre-built item.</summary>
        public SafeTensorBuilder AddTensor(TensorItem tensor)
        {
            if (tensor is null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            _tensors.Add(tensor);
            return this;
        }

        /// <summary>
        /// Writes the file, replacing any existing one only once the new file is complete.
        /// </summary>
        public void Save(string path) => SafeTensorWriter.WriteFile(path, _tensors, _metadata, _alignTo8Bytes);

        /// <summary>Writes to a stream.</summary>
        public void WriteTo(Stream destination) => SafeTensorWriter.Write(destination, _tensors, _metadata, _alignTo8Bytes);

        /// <summary>Serialises to a new byte array.</summary>
        public byte[] ToByteArray() => SafeTensorWriter.WriteToBytes(_tensors, _metadata, _alignTo8Bytes);
    }
}
