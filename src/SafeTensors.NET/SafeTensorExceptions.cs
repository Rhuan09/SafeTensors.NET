using System;

namespace SafeTensors
{
    /// <summary>
    /// Base type for every error raised by SafeTensors.NET.
    /// </summary>
    /// <remarks>
    /// Every failure that originates inside this library derives from this type. A caller
    /// that wraps an <c>Open</c> or <c>Save</c> call in <c>catch (SafeTensorException)</c>
    /// will not see a stray <see cref="InvalidOperationException"/> leak out of the parser.
    /// I/O failures from the underlying stream or file system are the deliberate exception:
    /// those surface as the framework's own <see cref="System.IO.IOException"/> family.
    /// </remarks>
    public class SafeTensorException : Exception
    {
        /// <summary>Creates the exception with a message.</summary>
        public SafeTensorException(string message)
            : base(message)
        {
        }

        /// <summary>Creates the exception with a message and an inner cause.</summary>
        public SafeTensorException(string message, Exception innerException)
            : base(message, innerException)
        {
        }
    }

    /// <summary>
    /// The header could not be decoded: bad length prefix, malformed JSON, or a field of
    /// the wrong shape or type.
    /// </summary>
    public class SafeTensorCorruptHeaderException : SafeTensorException
    {
        /// <summary>Creates the exception with a message.</summary>
        public SafeTensorCorruptHeaderException(string message)
            : base(message)
        {
        }

        /// <summary>Creates the exception with a message and an inner cause.</summary>
        public SafeTensorCorruptHeaderException(string message, Exception innerException)
            : base(message, innerException)
        {
        }
    }

    /// <summary>
    /// The header decoded, but describes a layout that is not internally consistent:
    /// a size that disagrees with the shape and dtype, tensors whose byte ranges overlap,
    /// or a range that runs past the end of the file.
    /// </summary>
    /// <remarks>
    /// Overlapping ranges are rejected rather than tolerated. A file where two tensors
    /// alias the same bytes is not a file any honest producer writes, and accepting it
    /// would let a crafted checkpoint hand the same memory to two different consumers.
    /// </remarks>
    public class SafeTensorValidationException : SafeTensorException
    {
        /// <summary>Creates the exception with a message.</summary>
        public SafeTensorValidationException(string message)
            : base(message)
        {
        }

        /// <summary>Creates the exception with a message and an inner cause.</summary>
        public SafeTensorValidationException(string message, Exception innerException)
            : base(message, innerException)
        {
        }
    }

    /// <summary>
    /// The requested tensor name is not present in the file or in the shard index.
    /// </summary>
    public class SafeTensorNotFoundException : SafeTensorException
    {
        /// <summary>Gets the tensor name that was not found.</summary>
        public string TensorName { get; }

        /// <summary>Creates the exception for a missing tensor name.</summary>
        public SafeTensorNotFoundException(string tensorName)
            : base($"Tensor '{tensorName}' was not found.")
        {
            TensorName = tensorName;
        }

        /// <summary>Creates the exception for a missing tensor name with a custom message.</summary>
        public SafeTensorNotFoundException(string tensorName, string message)
            : base(message)
        {
            TensorName = tensorName;
        }
    }

    /// <summary>
    /// A tensor is larger than the requested single-span view can address.
    /// </summary>
    /// <remarks>
    /// <see cref="Span{T}"/> is limited to <see cref="int.MaxValue"/> elements, so a tensor
    /// of 2 GiB or more cannot be handed out as one span. This is a real case in modern
    /// checkpoints — an fp32 embedding matrix crosses it easily. Use
    /// <see cref="TensorView.AsSpan{T}(long, int)"/> to walk the tensor in windows, or
    /// <see cref="TensorView.OpenStream"/> to read it sequentially.
    /// </remarks>
    public class TensorTooLargeException : SafeTensorException
    {
        /// <summary>Gets the tensor name.</summary>
        public string TensorName { get; }

        /// <summary>Gets the tensor length in bytes.</summary>
        public long ByteLength { get; }

        /// <summary>Creates the exception.</summary>
        public TensorTooLargeException(string tensorName, long byteLength, string message)
            : base(message)
        {
            TensorName = tensorName;
            ByteLength = byteLength;
        }
    }
}
