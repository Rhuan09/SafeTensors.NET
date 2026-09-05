using System;
using System.Buffers;
using System.Runtime.InteropServices;

namespace SafeTensors.Internal
{
    /// <summary>
    /// Reinterprets a <see cref="ReadOnlyMemory{T}"/> of bytes as a memory of
    /// <typeparamref name="T"/>.
    /// </summary>
    /// <remarks>
    /// <see cref="MemoryMarshal.Cast{TFrom, TTo}(ReadOnlySpan{TFrom})"/> only works on spans;
    /// <see cref="Memory{T}"/> has no equivalent, because the reinterpretation has to survive
    /// every later call to <c>.Span</c>. Wrapping the source memory preserves whatever
    /// lifetime checks it already performs, which matters because the source here is usually
    /// a memory mapping that throws once the file is disposed.
    /// </remarks>
    internal sealed class CastMemoryManager<T> : MemoryManager<T>
        where T : unmanaged
    {
        private readonly ReadOnlyMemory<byte> _source;
        private MemoryHandle _sourceHandle;
        private bool _pinned;

        public CastMemoryManager(ReadOnlyMemory<byte> source) => _source = source;

        public override Span<T> GetSpan()
        {
            // AsMemory re-exposes the same memory as writable so the cast has a Span<T> to
            // produce; taking .Span on it still runs the source's bounds and disposal
            // checks first. Only ReadOnlyMemory<T> is ever handed to callers.
            return MemoryMarshal.Cast<byte, T>(MemoryMarshal.AsMemory(_source).Span);
        }

        public override unsafe MemoryHandle Pin(int elementIndex = 0)
        {
            Span<T> span = GetSpan();
            if ((uint)elementIndex > (uint)span.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(elementIndex));
            }

            _sourceHandle = _source.Pin();
            _pinned = true;
            return new MemoryHandle((byte*)_sourceHandle.Pointer + ((long)elementIndex * sizeof(T)), default, this);
        }

        public override void Unpin()
        {
            if (_pinned)
            {
                _pinned = false;
                _sourceHandle.Dispose();
            }
        }

        protected override void Dispose(bool disposing) => Unpin();
    }
}
