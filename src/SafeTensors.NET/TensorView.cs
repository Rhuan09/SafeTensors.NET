using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using SafeTensors.Internal;

namespace SafeTensors
{
    /// <summary>
    /// A single tensor inside an open SafeTensors file.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A view holds no data of its own. It is a name, a shape and a byte range pointing into
    /// whatever the file was opened over, so obtaining one costs nothing and reading from it
    /// costs only what the storage costs.
    /// </para>
    /// <para><b>Lifetime.</b> When the file is memory mapped, <see cref="AsSpan{T}()"/>
    /// returns a span pointing at mapped pages. Those pages are unmapped when the owning
    /// <see cref="SafeTensorFile"/> is disposed, and a span that outlives the file reads
    /// memory the process no longer owns — which is an access violation, not an exception
    /// you can catch. The library cannot express that constraint in the type system, so it
    /// gives you two ways to avoid needing to: <see cref="AsMemory{T}()"/> re-checks the
    /// file on every access and throws <see cref="ObjectDisposedException"/> instead, and
    /// <see cref="ToArray{T}()"/> gives you a copy that owns itself. Use a span inside the
    /// scope of the <c>using</c> that owns the file; use memory or an array to escape it.
    /// </para>
    /// </remarks>
    public sealed class TensorView
    {
        private readonly ITensorDataSource _source;

        /// <summary>Gets the header entry describing this tensor.</summary>
        public TensorMetadata Metadata { get; }

        /// <summary>Gets the tensor name.</summary>
        public string Name => Metadata.Name;

        /// <summary>Gets the element type.</summary>
        public SafeTensorDType DType => Metadata.DType;

        /// <summary>Gets the dimensions.</summary>
        public IReadOnlyList<long> Shape => Metadata.Shape;

        /// <summary>Gets the length in bytes.</summary>
        public long ByteLength => Metadata.ByteLength;

        /// <summary>Gets the number of elements.</summary>
        public long ElementCount => Metadata.ElementCount;

        /// <summary>Gets the rank.</summary>
        public int Rank => Metadata.Rank;

        /// <summary>
        /// Gets a value indicating whether reads come straight from storage with no copy.
        /// True for memory-mapped and in-memory files, false when reading from a stream.
        /// </summary>
        public bool IsZeroCopy => _source.IsZeroCopy;

        /// <summary>
        /// Gets a value indicating whether the tensor fits in a single span, that is whether
        /// it is smaller than 2 GiB.
        /// </summary>
        public bool FitsInSingleSpan => ByteLength <= int.MaxValue;

        internal TensorView(TensorMetadata metadata, ITensorDataSource source)
        {
            Metadata = metadata ?? throw new ArgumentNullException(nameof(metadata));
            _source = source ?? throw new ArgumentNullException(nameof(source));
        }

        /// <summary>
        /// Gets the whole tensor as raw bytes.
        /// </summary>
        /// <exception cref="TensorTooLargeException">The tensor is 2 GiB or larger.</exception>
        /// <exception cref="ObjectDisposedException">The owning file has been disposed.</exception>
        public ReadOnlySpan<byte> AsSpan()
        {
            RequireSingleSpan();
            return _source.GetSpan(Metadata.DataStart, (int)ByteLength);
        }

        /// <summary>
        /// Gets the whole tensor reinterpreted as <typeparamref name="T"/>.
        /// </summary>
        /// <typeparam name="T">
        /// An element type whose size divides the tensor length. It is not checked against
        /// <see cref="DType"/>: reading an F32 tensor as <see cref="uint"/> to inspect bit
        /// patterns is legitimate, and forbidding it would buy nothing.
        /// </typeparam>
        /// <exception cref="TensorTooLargeException">The tensor is 2 GiB or larger.</exception>
        public ReadOnlySpan<T> AsSpan<T>()
            where T : unmanaged
            => MemoryMarshal.Cast<byte, T>(AsSpan());

        /// <summary>
        /// Gets a window of <paramref name="count"/> elements starting at
        /// <paramref name="elementOffset"/>.
        /// </summary>
        /// <remarks>
        /// This is the way to read a tensor too large for a single span. An fp32 embedding
        /// matrix of a large model crosses 2 GiB comfortably, and no amount of API design
        /// makes <see cref="Span{T}"/> address more than <see cref="int.MaxValue"/> elements.
        /// </remarks>
        public ReadOnlySpan<T> AsSpan<T>(long elementOffset, int count)
            where T : unmanaged
        {
            if (elementOffset < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(elementOffset));
            }

            if (count < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(count));
            }

            int elementSize = SizeOf<T>();
            long byteOffset = checked(elementOffset * elementSize);
            long byteCount = checked((long)count * elementSize);

            if (byteOffset > ByteLength - byteCount)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(count),
                    $"Window [{elementOffset}, {elementOffset + count}) runs past the end of " +
                    $"tensor '{Name}', which holds {ByteLength / elementSize} elements of this size.");
            }

            return MemoryMarshal.Cast<byte, T>(
                _source.GetSpan(Metadata.DataStart + byteOffset, (int)byteCount));
        }

        /// <summary>
        /// Gets the tensor as memory that stays valid to hold across calls, and throws
        /// <see cref="ObjectDisposedException"/> rather than reading freed pages if the
        /// owning file is disposed first.
        /// </summary>
        /// <remarks>
        /// This is the accessor to reach for when the tensor has to outlive the expression
        /// that produced it — stored in a field, captured by a lambda, passed to an async
        /// method. It is the same bytes as <see cref="AsSpan()"/> with a lifetime check on
        /// every access.
        /// </remarks>
        /// <exception cref="TensorTooLargeException">The tensor is 2 GiB or larger.</exception>
        public ReadOnlyMemory<byte> AsMemory()
        {
            RequireSingleSpan();
            return _source.GetMemory(Metadata.DataStart, (int)ByteLength);
        }

        /// <summary>
        /// Gets the tensor as typed memory that is safe to hold across calls.
        /// </summary>
        /// <exception cref="TensorTooLargeException">The tensor is 2 GiB or larger.</exception>
        public ReadOnlyMemory<T> AsMemory<T>()
            where T : unmanaged
            => new CastMemoryManager<T>(AsMemory()).Memory;

        /// <summary>
        /// Gets a pointer to the first byte, or <c>null</c> when the storage has no stable
        /// address.
        /// </summary>
        /// <remarks>
        /// Only a memory-mapped file returns a pointer; in-memory and stream-backed files
        /// return <c>null</c> because managed buffers move and streams have no address. The
        /// pointer is valid until the owning <see cref="SafeTensorFile"/> is disposed and
        /// not one instruction longer, which is why the name says so. This is the handoff
        /// point for GPU uploads and native interop.
        /// </remarks>
        public unsafe void* DangerousGetPointer() => _source.TryGetPointer(Metadata.DataStart);

        /// <summary>
        /// Copies the tensor into a new array.
        /// </summary>
        /// <exception cref="TensorTooLargeException">The tensor is 2 GiB or larger.</exception>
        public byte[] ToArray()
        {
            RequireSingleSpan();
            byte[] result = new byte[ByteLength];
            _source.CopyTo(Metadata.DataStart, result);
            return result;
        }

        /// <summary>
        /// Copies the tensor into a new typed array.
        /// </summary>
        /// <exception cref="TensorTooLargeException">The tensor is 2 GiB or larger.</exception>
        public T[] ToArray<T>()
            where T : unmanaged
        {
            RequireSingleSpan();

            int elementSize = SizeOf<T>();
            if (ByteLength % elementSize != 0)
            {
                throw new ArgumentException(
                    $"Tensor '{Name}' is {ByteLength} bytes, which is not a whole number of " +
                    $"{elementSize}-byte elements.");
            }

            T[] result = new T[ByteLength / elementSize];
            _source.CopyTo(Metadata.DataStart, MemoryMarshal.AsBytes(result.AsSpan()));
            return result;
        }

        /// <summary>Copies the tensor bytes into <paramref name="destination"/>.</summary>
        public void CopyTo(Span<byte> destination)
        {
            if (destination.Length < ByteLength)
            {
                throw new ArgumentException(
                    $"Destination holds {destination.Length} bytes but tensor '{Name}' needs {ByteLength}.",
                    nameof(destination));
            }

            RequireSingleSpan();
            _source.CopyTo(Metadata.DataStart, destination.Slice(0, (int)ByteLength));
        }

        /// <summary>Copies the tensor into a typed destination.</summary>
        public void CopyTo<T>(Span<T> destination)
            where T : unmanaged
            => CopyTo(MemoryMarshal.AsBytes(destination));

        /// <summary>
        /// Asynchronously copies the tensor bytes into <paramref name="destination"/>.
        /// </summary>
        /// <remarks>
        /// Only meaningful for stream-backed files; memory-mapped and in-memory sources
        /// complete synchronously because there is no I/O to await.
        /// </remarks>
        public ValueTask CopyToAsync(Memory<byte> destination, CancellationToken cancellationToken = default)
        {
            if (destination.Length < ByteLength)
            {
                throw new ArgumentException(
                    $"Destination holds {destination.Length} bytes but tensor '{Name}' needs {ByteLength}.",
                    nameof(destination));
            }

            RequireSingleSpan();
            return _source.CopyToAsync(Metadata.DataStart, destination.Slice(0, (int)ByteLength), cancellationToken);
        }

        /// <summary>
        /// Opens a read-only stream over the tensor bytes. Works at any size.
        /// </summary>
        public Stream OpenStream() => _source.OpenStream(Metadata.DataStart, ByteLength);

        /// <summary>
        /// Takes a contiguous run of the outermost dimension as a tensor in its own right,
        /// without copying.
        /// </summary>
        /// <param name="start">First index along dimension 0.</param>
        /// <param name="count">How many indices to take.</param>
        /// <remarks>
        /// Tensors are stored in row-major order, so a range of the outermost dimension is
        /// one contiguous byte range and can be handed out as a view. This is what makes it
        /// possible to pull 32 rows out of a 250 000-row embedding matrix while touching
        /// only the pages those rows live on. Slicing an inner dimension would be strided
        /// rather than contiguous and is deliberately not offered as a "view".
        /// </remarks>
        public TensorView Slice(long start, long count)
        {
            if (Rank == 0)
            {
                throw new InvalidOperationException($"Tensor '{Name}' is a scalar and has no dimension to slice.");
            }

            long outer = Metadata.Shape[0];
            if (start < 0 || count < 0 || start > outer - count)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(start),
                    $"Range [{start}, {start + count}) is outside dimension 0 of '{Name}', which is {outer}.");
            }

            long rowElements = 1;
            for (int i = 1; i < Rank; i++)
            {
                rowElements = checked(rowElements * Metadata.Shape[i]);
            }

            long rowBytes = DTypes.ByteLength(DType, rowElements);
            long sliceStart = checked(Metadata.DataStart + (start * rowBytes));
            long sliceEnd = checked(sliceStart + (count * rowBytes));

            long[] shape = Metadata.ToShapeArray();
            shape[0] = count;

            var sliced = new TensorMetadata(
                $"{Name}[{start}:{start + count}]",
                DType,
                shape,
                sliceStart,
                sliceEnd);

            return new TensorView(sliced, _source);
        }

        /// <summary>
        /// Reads the tensor as <see cref="float"/>, converting from whichever numeric dtype
        /// it actually holds.
        /// </summary>
        /// <remarks>
        /// The conversion is a real cost: it allocates the full array and widens every
        /// element. For F32 data prefer <see cref="AsSpan{T}()"/>, which does neither.
        /// </remarks>
        public float[] ToSingleArray()
        {
            if (ElementCount > int.MaxValue)
            {
                throw new TensorTooLargeException(
                    Name,
                    ByteLength,
                    $"Tensor '{Name}' holds {ElementCount} elements, more than an array can address. " +
                    "Read it in windows with AsSpan<T>(offset, count).");
            }

            int count = (int)ElementCount;
            float[] result = new float[count];

            switch (DType)
            {
                case SafeTensorDType.F32:
                    AsSpan<float>().CopyTo(result);
                    break;
                case SafeTensorDType.F16:
                    Widen(AsSpan<Float16>(), result, static v => v);
                    break;
                case SafeTensorDType.BF16:
                    Widen(AsSpan<BFloat16>(), result, static v => v);
                    break;
                case SafeTensorDType.F64:
                    Widen(AsSpan<double>(), result, static v => (float)v);
                    break;
                case SafeTensorDType.I64:
                    Widen(AsSpan<long>(), result, static v => v);
                    break;
                case SafeTensorDType.U64:
                    Widen(AsSpan<ulong>(), result, static v => v);
                    break;
                case SafeTensorDType.I32:
                    Widen(AsSpan<int>(), result, static v => v);
                    break;
                case SafeTensorDType.U32:
                    Widen(AsSpan<uint>(), result, static v => v);
                    break;
                case SafeTensorDType.I16:
                    Widen(AsSpan<short>(), result, static v => v);
                    break;
                case SafeTensorDType.U16:
                    Widen(AsSpan<ushort>(), result, static v => v);
                    break;
                case SafeTensorDType.I8:
                    Widen(AsSpan<sbyte>(), result, static v => v);
                    break;
                case SafeTensorDType.U8:
                    Widen(AsSpan<byte>(), result, static v => v);
                    break;
                case SafeTensorDType.BOOL:
                    Widen(AsSpan<byte>(), result, static v => v != 0 ? 1f : 0f);
                    break;
                default:
                    throw new NotSupportedException(
                        $"Tensor '{Name}' has dtype {DType}, which has no defined conversion to float. " +
                        "Read its raw bytes with AsSpan() and decode them yourself.");
            }

            return result;
        }

        private static void Widen<T>(ReadOnlySpan<T> source, float[] destination, Func<T, float> convert)
            where T : unmanaged
        {
            for (int i = 0; i < destination.Length; i++)
            {
                destination[i] = convert(source[i]);
            }
        }

        private static int SizeOf<T>()
            where T : unmanaged
        {
            unsafe
            {
                return sizeof(T);
            }
        }

        private void RequireSingleSpan()
        {
            if (ByteLength > int.MaxValue)
            {
                throw new TensorTooLargeException(
                    Name,
                    ByteLength,
                    $"Tensor '{Name}' is {ByteLength} bytes, more than a single Span or array can " +
                    "address. Read it in windows with AsSpan<T>(offset, count), stream it with " +
                    "OpenStream(), or take a Slice() of the outermost dimension.");
            }
        }

        /// <inheritdoc />
        public override string ToString() => Metadata.ToString();
    }
}
