using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace SafeTensors.Internal
{
    /// <summary>
    /// Tensor bytes served from a buffer that is already in managed memory.
    /// </summary>
    internal sealed class MemoryDataSource : ITensorDataSource
    {
        private readonly ReadOnlyMemory<byte> _buffer;
        private readonly int _dataOffset;
        private readonly int _dataLength;
        private int _disposed;

        public bool IsZeroCopy => true;

        public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

        public MemoryDataSource(ReadOnlyMemory<byte> buffer, long dataOffset, long dataLength)
        {
            // A ReadOnlyMemory<byte> cannot exceed int.MaxValue, so these casts are safe by
            // construction: the caller already bounded them against buffer.Length.
            _buffer = buffer;
            _dataOffset = checked((int)dataOffset);
            _dataLength = checked((int)dataLength);
        }

        public unsafe byte* TryGetPointer(long offset)
        {
            // Managed memory moves. Handing out a bare pointer without a pin would be a
            // promise this source cannot keep, so it declines instead.
            return null;
        }

        public ReadOnlySpan<byte> GetSpan(long offset, int length)
        {
            ThrowIfDisposed();
            ValidateRange(offset, length);
            return _buffer.Span.Slice(_dataOffset + (int)offset, length);
        }

        public ReadOnlyMemory<byte> GetMemory(long offset, int length)
        {
            ThrowIfDisposed();
            ValidateRange(offset, length);
            return _buffer.Slice(_dataOffset + (int)offset, length);
        }

        public void CopyTo(long offset, Span<byte> destination)
        {
            GetSpan(offset, destination.Length).CopyTo(destination);
        }

        public ValueTask CopyToAsync(long offset, Memory<byte> destination, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            CopyTo(offset, destination.Span);
            return default;
        }

        public Stream OpenStream(long offset, long length)
        {
            ThrowIfDisposed();
            ValidateRange(offset, length);

            ReadOnlyMemory<byte> slice = _buffer.Slice(_dataOffset + (int)offset, (int)length);

            // Prefer a view over the existing buffer; only copy when the memory is not
            // array-backed and there is nothing to view.
            if (MemoryMarshal.TryGetArray(slice, out ArraySegment<byte> segment) && segment.Array is not null)
            {
                return new MemoryStream(segment.Array, segment.Offset, segment.Count, writable: false);
            }

            return new MemoryStream(slice.ToArray(), writable: false);
        }

        private void ValidateRange(long offset, long length)
        {
            if (offset < 0 || length < 0 || offset > _dataLength - length)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(offset),
                    $"Range [{offset}, {offset + length}) lies outside the {_dataLength}-byte data section.");
            }
        }

        private void ThrowIfDisposed()
        {
            if (IsDisposed)
            {
                throw new ObjectDisposedException(nameof(SafeTensorFile));
            }
        }

        public void Dispose() => Interlocked.Exchange(ref _disposed, 1);
    }
}
