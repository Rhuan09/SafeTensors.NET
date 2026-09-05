using System;
using System.Collections.Generic;

namespace SafeTensors
{
    /// <summary>
    /// One tensor's entry in the header: its name, element type, shape, and the byte range
    /// it occupies inside the data section.
    /// </summary>
    /// <remarks>
    /// Offsets are relative to the start of the data section, not to the start of the file.
    /// Add <see cref="SafeTensorHeader.DataOffset"/> to get a file offset.
    /// </remarks>
    public sealed class TensorMetadata
    {
        private readonly long[] _shape;

        /// <summary>Gets the tensor name, for example <c>model.layers.0.mlp.up_proj.weight</c>.</summary>
        public string Name { get; }

        /// <summary>Gets the element type.</summary>
        public SafeTensorDType DType { get; }

        /// <summary>Gets the dimensions. An empty shape denotes a scalar.</summary>
        public IReadOnlyList<long> Shape => _shape;

        /// <summary>Gets the dimensions without an interface dispatch.</summary>
        public ReadOnlySpan<long> ShapeSpan => _shape;

        /// <summary>Gets the rank, that is the number of dimensions.</summary>
        public int Rank => _shape.Length;

        /// <summary>Gets the first byte of this tensor, relative to the data section.</summary>
        public long DataStart { get; }

        /// <summary>Gets the byte after the last, relative to the data section.</summary>
        public long DataEnd { get; }

        /// <summary>Gets the length in bytes.</summary>
        public long ByteLength => DataEnd - DataStart;

        /// <summary>Gets the number of elements. A scalar has one element, not zero.</summary>
        public long ElementCount { get; }

        /// <summary>
        /// Creates tensor metadata and checks that the shape, dtype and byte range agree.
        /// </summary>
        /// <exception cref="SafeTensorValidationException">
        /// The shape overflows, or the byte range does not match the size the shape and
        /// dtype imply.
        /// </exception>
        public TensorMetadata(string name, SafeTensorDType dtype, long[] shape, long dataStart, long dataEnd)
        {
            if (string.IsNullOrEmpty(name))
            {
                throw new ArgumentException("Tensor name cannot be null or empty.", nameof(name));
            }

            if (shape is null)
            {
                throw new ArgumentNullException(nameof(shape));
            }

            if (dataStart < 0)
            {
                throw new SafeTensorValidationException(
                    $"Tensor '{name}' has a negative start offset ({dataStart}).");
            }

            if (dataEnd < dataStart)
            {
                throw new SafeTensorValidationException(
                    $"Tensor '{name}' ends before it starts (offsets [{dataStart}, {dataEnd}]).");
            }

            Name = name;
            DType = dtype;
            _shape = (long[])shape.Clone();
            DataStart = dataStart;
            DataEnd = dataEnd;
            ElementCount = ComputeElementCount(name, _shape);

            long expected = DTypes.ByteLength(dtype, ElementCount);
            if (expected != ByteLength)
            {
                throw new SafeTensorValidationException(
                    $"Tensor '{name}' declares shape [{string.Join(", ", _shape)}] of {dtype}, " +
                    $"which is {expected} bytes, but its offsets [{dataStart}, {dataEnd}] " +
                    $"cover {ByteLength} bytes.");
            }
        }

        private static long ComputeElementCount(string name, long[] shape)
        {
            long count = 1;
            for (int i = 0; i < shape.Length; i++)
            {
                if (shape[i] < 0)
                {
                    throw new SafeTensorValidationException(
                        $"Tensor '{name}' has a negative dimension at index {i}: {shape[i]}.");
                }

                try
                {
                    count = checked(count * shape[i]);
                }
                catch (OverflowException ex)
                {
                    // A crafted header can multiply out past 2^63 and, without this, wrap to
                    // a small positive number that then agrees with a small byte range.
                    throw new SafeTensorValidationException(
                        $"Tensor '{name}' has a shape whose element count overflows 64 bits: " +
                        $"[{string.Join(", ", shape)}].", ex);
                }
            }

            return count;
        }

        /// <summary>Returns a copy of the shape as an array.</summary>
        public long[] ToShapeArray() => (long[])_shape.Clone();

        /// <inheritdoc />
        public override string ToString()
            => $"{Name} [{DType} ({string.Join(", ", _shape)}), {ByteLength} bytes at [{DataStart}, {DataEnd})]";
    }
}
