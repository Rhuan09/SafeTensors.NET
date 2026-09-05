using System;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Threading;
using System.Threading.Tasks;

namespace SafeTensors.Internal
{
    /// <summary>
    /// Tensor bytes served straight out of a read-only memory mapping.
    /// </summary>
    /// <remarks>
    /// This is the path that makes "zero copy" true rather than a slogan: the operating
    /// system pages the file in on demand and a span points at those pages, so opening a
    /// 40 GB checkpoint costs the header parse and nothing else.
    /// </remarks>
    internal sealed unsafe class MemoryMappedDataSource : ITensorDataSource
    {
        private readonly FileStream? _file;
        private readonly MemoryMappedFile _mapping;
        private readonly MemoryMappedViewAccessor _accessor;
        private readonly byte* _basePointer;
        private readonly long _dataOffset;
        private readonly long _dataLength;
        private int _disposed;

        public bool IsZeroCopy => true;

        public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

        public MemoryMappedDataSource(
            FileStream? file,
            MemoryMappedFile mapping,
            MemoryMappedViewAccessor accessor,
            long dataOffset,
            long dataLength)
        {
            _file = file;
            _mapping = mapping;
            _accessor = accessor;
            _dataOffset = dataOffset;
            _dataLength = dataLength;

            byte* pointer = null;
            _accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
            _basePointer = pointer;
        }

        public byte* TryGetPointer(long offset)
        {
            ThrowIfDisposed();
            ValidateRange(offset, 0);
            return _basePointer + _dataOffset + offset;
        }

        public ReadOnlySpan<byte> GetSpan(long offset, int length)
        {
            ThrowIfDisposed();
            ValidateRange(offset, length);
            return new ReadOnlySpan<byte>(_basePointer + _dataOffset + offset, length);
        }

        public ReadOnlyMemory<byte> GetMemory(long offset, int length)
        {
            ThrowIfDisposed();
            ValidateRange(offset, length);
            return new MappedMemoryManager(this, offset, length).Memory;
        }

        public void CopyTo(long offset, Span<byte> destination)
        {
            GetSpan(offset, destination.Length).CopyTo(destination);
        }

        public ValueTask CopyToAsync(long offset, Memory<byte> destination, CancellationToken cancellationToken)
        {
            // Paging is the kernel's job and there is no async form of a page fault. Copying
            // synchronously is honest; pushing it to the thread pool would only add latency.
            cancellationToken.ThrowIfCancellationRequested();
            CopyTo(offset, destination.Span);
            return default;
        }

        public Stream OpenStream(long offset, long length)
        {
            ThrowIfDisposed();
            ValidateRange(offset, length);
            return _mapping.CreateViewStream(_dataOffset + offset, length, MemoryMappedFileAccess.Read);
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
                throw new ObjectDisposedException(
                    nameof(SafeTensorFile),
                    "The SafeTensors file has been disposed and its mapping released.");
            }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
            {
                return;
            }

            if (_basePointer != null)
            {
                _accessor.SafeMemoryMappedViewHandle.ReleasePointer();
            }

            _accessor.Dispose();
            _mapping.Dispose();
            _file?.Dispose();
        }

        /// <summary>
        /// Wraps a mapped range as <see cref="ReadOnlyMemory{T}"/> and re-checks the
        /// mapping every time the span is taken.
        /// </summary>
        /// <remarks>
        /// This is what makes <see cref="TensorView.AsMemory{T}"/> safe to store in a field.
        /// A raw span captured from a mapping and used after the file is disposed reads
        /// unmapped pages and takes the process down; going through a memory manager turns
        /// that into an <see cref="ObjectDisposedException"/>.
        /// </remarks>
        private sealed class MappedMemoryManager : System.Buffers.MemoryManager<byte>
        {
            private readonly MemoryMappedDataSource _source;
            private readonly long _offset;
            private readonly int _length;

            public MappedMemoryManager(MemoryMappedDataSource source, long offset, int length)
            {
                _source = source;
                _offset = offset;
                _length = length;
            }

            public override Span<byte> GetSpan()
            {
                _source.ThrowIfDisposed();
                return new Span<byte>(_source._basePointer + _source._dataOffset + _offset, _length);
            }

            public override System.Buffers.MemoryHandle Pin(int elementIndex = 0)
            {
                if (elementIndex < 0 || elementIndex > _length)
                {
                    throw new ArgumentOutOfRangeException(nameof(elementIndex));
                }

                _source.ThrowIfDisposed();

                // Mapped pages do not move, so there is nothing to pin; the handle exists
                // only so callers can take a pointer.
                return new System.Buffers.MemoryHandle(_source._basePointer + _source._dataOffset + _offset + elementIndex);
            }

            public override void Unpin()
            {
            }

            protected override void Dispose(bool disposing)
            {
            }
        }
    }
}
