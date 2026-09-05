using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace SafeTensors.Internal
{
    /// <summary>
    /// Tensor bytes read on demand from a seekable stream.
    /// </summary>
    /// <remarks>
    /// This source is not zero-copy: every read seeks and copies. It exists so that a file
    /// arriving over a network, out of an archive, or from a stream you do not own can be
    /// read without staging the whole thing in memory first. Access is serialised on one
    /// lock because a stream has a single shared position.
    /// </remarks>
    internal sealed class StreamDataSource : ITensorDataSource
    {
        private readonly Stream _stream;
        private readonly long _dataOffset;
        private readonly long _dataLength;
        private readonly bool _leaveOpen;
        private readonly SemaphoreSlim _gate = new SemaphoreSlim(1, 1);
        private int _disposed;

        public bool IsZeroCopy => false;

        public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

        public StreamDataSource(Stream stream, long dataOffset, long dataLength, bool leaveOpen)
        {
            if (!stream.CanSeek)
            {
                throw new ArgumentException("Random tensor access needs a seekable stream.", nameof(stream));
            }

            _stream = stream;
            _dataOffset = dataOffset;
            _dataLength = dataLength;
            _leaveOpen = leaveOpen;
        }

        public unsafe byte* TryGetPointer(long offset) => null;

        public ReadOnlySpan<byte> GetSpan(long offset, int length)
        {
            // The bytes are not in memory yet, so this materialises them. The array is
            // freshly allocated per call and never reused, which is why the returned span
            // stays valid after the source is disposed.
            byte[] buffer = new byte[length];
            CopyTo(offset, buffer);
            return buffer;
        }

        public ReadOnlyMemory<byte> GetMemory(long offset, int length)
        {
            byte[] buffer = new byte[length];
            CopyTo(offset, buffer);
            return buffer;
        }

        public void CopyTo(long offset, Span<byte> destination)
        {
            ThrowIfDisposed();
            ValidateRange(offset, destination.Length);

            _gate.Wait();
            try
            {
                _stream.Seek(_dataOffset + offset, SeekOrigin.Begin);
                BinaryUtils.ReadExactly(_stream, destination, "tensor data");
            }
            finally
            {
                _gate.Release();
            }
        }

        public async ValueTask CopyToAsync(long offset, Memory<byte> destination, CancellationToken cancellationToken)
        {
            ThrowIfDisposed();
            ValidateRange(offset, destination.Length);

            await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                _stream.Seek(_dataOffset + offset, SeekOrigin.Begin);

                int total = 0;
                while (total < destination.Length)
                {
#if NETSTANDARD2_0
                    byte[] rented = System.Buffers.ArrayPool<byte>.Shared.Rent(destination.Length - total);
                    try
                    {
                        int read = await _stream.ReadAsync(rented, 0, destination.Length - total, cancellationToken)
                            .ConfigureAwait(false);
                        if (read == 0)
                        {
                            throw new SafeTensorCorruptHeaderException(
                                $"Unexpected end of stream while reading tensor data: got {total} of {destination.Length} bytes.");
                        }

                        new ReadOnlySpan<byte>(rented, 0, read).CopyTo(destination.Span.Slice(total));
                        total += read;
                    }
                    finally
                    {
                        System.Buffers.ArrayPool<byte>.Shared.Return(rented);
                    }
#else
                    int read = await _stream.ReadAsync(destination.Slice(total), cancellationToken)
                        .ConfigureAwait(false);
                    if (read == 0)
                    {
                        throw new SafeTensorCorruptHeaderException(
                            $"Unexpected end of stream while reading tensor data: got {total} of {destination.Length} bytes.");
                    }

                    total += read;
#endif
                }
            }
            finally
            {
                _gate.Release();
            }
        }

        public Stream OpenStream(long offset, long length)
        {
            ThrowIfDisposed();
            ValidateRange(offset, length);
            return new RangeStream(this, _dataOffset + offset, length);
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

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
            {
                return;
            }

            if (!_leaveOpen)
            {
                _stream.Dispose();
            }

            _gate.Dispose();
        }

        /// <summary>
        /// A read-only window onto the parent stream that takes the parent's lock for the
        /// duration of each read, so several tensor streams can be alive at once.
        /// </summary>
        private sealed class RangeStream : Stream
        {
            private readonly StreamDataSource _owner;
            private readonly long _start;
            private readonly long _length;
            private long _position;

            public RangeStream(StreamDataSource owner, long start, long length)
            {
                _owner = owner;
                _start = start;
                _length = length;
            }

            public override bool CanRead => true;

            public override bool CanSeek => true;

            public override bool CanWrite => false;

            public override long Length => _length;

            public override long Position
            {
                get => _position;
                set
                {
                    if (value < 0 || value > _length)
                    {
                        throw new ArgumentOutOfRangeException(nameof(value));
                    }

                    _position = value;
                }
            }

            public override void Flush()
            {
            }

            public override int Read(byte[] buffer, int offset, int count)
            {
                if (buffer is null)
                {
                    throw new ArgumentNullException(nameof(buffer));
                }

                if (offset < 0 || count < 0 || offset > buffer.Length - count)
                {
                    throw new ArgumentOutOfRangeException(nameof(count));
                }

                if (_position >= _length || count == 0)
                {
                    return 0;
                }

                int toRead = (int)Math.Min(count, _length - _position);

                _owner.ThrowIfDisposed();
                _owner._gate.Wait();
                try
                {
                    _owner._stream.Seek(_start + _position, SeekOrigin.Begin);
                    int read = _owner._stream.Read(buffer, offset, toRead);
                    _position += read;
                    return read;
                }
                finally
                {
                    _owner._gate.Release();
                }
            }

            public override long Seek(long offset, SeekOrigin origin)
            {
                long target = origin switch
                {
                    SeekOrigin.Begin => offset,
                    SeekOrigin.Current => _position + offset,
                    SeekOrigin.End => _length + offset,
                    _ => throw new ArgumentOutOfRangeException(nameof(origin))
                };

                if (target < 0 || target > _length)
                {
                    throw new IOException("Seek target is outside the tensor range.");
                }

                _position = target;
                return _position;
            }

            public override void SetLength(long value) => throw new NotSupportedException();

            public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        }
    }
}
