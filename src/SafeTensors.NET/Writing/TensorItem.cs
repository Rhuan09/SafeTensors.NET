using System;
using System.IO;
using System.Runtime.InteropServices;
using SafeTensors.Internal;

namespace SafeTensors
{
    /// <summary>
    /// One tensor queued for writing: a name, a dtype, a shape, and where its bytes come
    /// from.
    /// </summary>
    /// <remarks>
    /// <para>
    /// An item created from an array or a memory keeps a <b>reference</b> to it, not a copy.
    /// Writing a checkpoint would otherwise cost a second full copy of the model, which for
    /// the sizes this library exists to handle is not a detail. The consequence is the usual
    /// one for builders: do not mutate the source between adding a tensor and writing the
    /// file.
    /// </para>
    /// <para>
    /// The reserved name <c>__metadata__</c> is rejected, because a tensor by that name
    /// would collide with the metadata object in the header and produce a file that other
    /// readers decode differently.
    /// </para>
    /// </remarks>
    public sealed class TensorItem
    {
        /// <summary>The header key reserved for file metadata.</summary>
        internal const string MetadataKey = "__metadata__";

        private readonly ReadOnlyMemory<byte> _bytes;
        private readonly Action<Stream>? _writer;

        /// <summary>Gets the tensor name.</summary>
        public string Name { get; }

        /// <summary>Gets the element type.</summary>
        public SafeTensorDType DType { get; }

        /// <summary>Gets the dimensions.</summary>
        public long[] Shape { get; }

        /// <summary>Gets the length in bytes.</summary>
        public long ByteLength { get; }

        /// <summary>Gets the number of elements.</summary>
        public long ElementCount { get; }

        /// <summary>Creates an item whose bytes are already in memory.</summary>
        public TensorItem(string name, SafeTensorDType dtype, long[] shape, ReadOnlyMemory<byte> data)
            : this(name, dtype, shape, data.Length, writer: null, validate: true)
        {
            _bytes = data;
        }

        /// <summary>
        /// Creates an item that produces its bytes on demand.
        /// </summary>
        /// <param name="name">Tensor name.</param>
        /// <param name="dtype">Element type.</param>
        /// <param name="shape">Dimensions.</param>
        /// <param name="byteLength">Exactly how many bytes <paramref name="writer"/> will write.</param>
        /// <param name="writer">Writes the tensor payload to the destination stream.</param>
        /// <remarks>
        /// This is the overload for data that does not exist as a buffer: generated weights,
        /// a tensor being copied out of another file, a decompressing reader. The declared
        /// length is checked against the shape and dtype up front, but nothing can verify
        /// the callback actually writes that many bytes until it runs.
        /// </remarks>
        public TensorItem(string name, SafeTensorDType dtype, long[] shape, long byteLength, Action<Stream> writer)
            : this(name, dtype, shape, byteLength, writer ?? throw new ArgumentNullException(nameof(writer)), validate: true)
        {
        }

        private TensorItem(
            string name,
            SafeTensorDType dtype,
            long[] shape,
            long byteLength,
            Action<Stream>? writer,
            bool validate)
        {
            if (string.IsNullOrEmpty(name))
            {
                throw new ArgumentException("Tensor name cannot be null or empty.", nameof(name));
            }

            if (string.Equals(name, MetadataKey, StringComparison.Ordinal))
            {
                throw new ArgumentException(
                    $"'{MetadataKey}' is reserved for file metadata and cannot be used as a tensor name.",
                    nameof(name));
            }

            if (shape is null)
            {
                throw new ArgumentNullException(nameof(shape));
            }

            if (byteLength < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(byteLength));
            }

            Name = name;
            DType = dtype;
            Shape = (long[])shape.Clone();
            ByteLength = byteLength;
            _writer = writer;

            long count = 1;
            for (int i = 0; i < Shape.Length; i++)
            {
                if (Shape[i] < 0)
                {
                    throw new SafeTensorValidationException(
                        $"Tensor '{name}' has a negative dimension at index {i}: {Shape[i]}.");
                }

                try
                {
                    count = checked(count * Shape[i]);
                }
                catch (OverflowException ex)
                {
                    throw new SafeTensorValidationException(
                        $"Tensor '{name}' has a shape whose element count overflows 64 bits: " +
                        $"[{string.Join(", ", Shape)}].", ex);
                }
            }

            ElementCount = count;

            if (!validate)
            {
                return;
            }

            long expected = DTypes.ByteLength(dtype, ElementCount);
            if (expected != ByteLength)
            {
                throw new SafeTensorValidationException(
                    $"Tensor '{name}' has {ByteLength} bytes of data, but shape " +
                    $"[{string.Join(", ", Shape)}] of {dtype} needs {expected}.");
            }
        }

        /// <summary>
        /// Creates an item from a typed array. The array is referenced, not copied.
        /// </summary>
        public static TensorItem FromArray<T>(string name, T[] array, long[] shape)
            where T : unmanaged
        {
            if (array is null)
            {
                throw new ArgumentNullException(nameof(array));
            }

            return FromMemory<T>(name, array, shape);
        }

        /// <summary>
        /// Creates an item from typed memory. The memory is referenced, not copied.
        /// </summary>
        public static TensorItem FromMemory<T>(string name, ReadOnlyMemory<T> data, long[] shape)
            where T : unmanaged
        {
            SafeTensorDType dtype = DTypes.FromClrType<T>();
            long byteLength;
            unsafe
            {
                byteLength = (long)data.Length * sizeof(T);
            }

            return new TensorItem(
                name,
                dtype,
                shape,
                byteLength,
                stream => StreamWrite.Span(stream, MemoryMarshal.AsBytes(data.Span)),
                validate: true);
        }

        /// <summary>
        /// Creates an item from a typed span. Unlike the array and memory overloads this
        /// one must copy, because a span cannot be stored.
        /// </summary>
        public static TensorItem FromSpan<T>(string name, ReadOnlySpan<T> data, long[] shape)
            where T : unmanaged
        {
            SafeTensorDType dtype = DTypes.FromClrType<T>();
            return new TensorItem(name, dtype, shape, MemoryMarshal.AsBytes(data).ToArray());
        }

        /// <summary>Creates an item from raw bytes with an explicit dtype.</summary>
        public static TensorItem FromBytes(string name, SafeTensorDType dtype, long[] shape, ReadOnlyMemory<byte> data)
            => new TensorItem(name, dtype, shape, data);

        /// <summary>Writes this tensor's payload to <paramref name="destination"/>.</summary>
        public void WriteTo(Stream destination)
        {
            if (destination is null)
            {
                throw new ArgumentNullException(nameof(destination));
            }

            if (_writer is not null)
            {
                _writer(destination);
                return;
            }

            StreamWrite.Span(destination, _bytes.Span);
        }

        /// <inheritdoc />
        public override string ToString()
            => $"{Name} [{DType} ({string.Join(", ", Shape)}), {ByteLength} bytes]";
    }
}
