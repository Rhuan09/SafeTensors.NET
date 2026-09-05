using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace SafeTensors.Internal
{
    /// <summary>
    /// Where a <see cref="TensorView"/> gets its bytes: a memory mapping, a buffer already
    /// in memory, or a seekable stream.
    /// </summary>
    /// <remarks>
    /// Deliberately internal. Exposing raw pointers on a public interface would commit the
    /// package to supporting third-party implementations of an unsafe contract forever, for
    /// the sake of a scenario nobody has asked for. Adding a source later is a
    /// non-breaking change; taking the interface back is not.
    /// </remarks>
    internal interface ITensorDataSource : IDisposable
    {
        /// <summary>
        /// Gets a value indicating whether spans come straight from the underlying storage
        /// with no copy.
        /// </summary>
        bool IsZeroCopy { get; }

        /// <summary>Gets a value indicating whether this source has been disposed.</summary>
        bool IsDisposed { get; }

        /// <summary>
        /// Gets a pointer to <paramref name="offset"/>, or <c>null</c> when this source is
        /// not backed by addressable memory.
        /// </summary>
        unsafe byte* TryGetPointer(long offset);

        /// <summary>Gets the bytes at <paramref name="offset"/>, copying only if it must.</summary>
        ReadOnlySpan<byte> GetSpan(long offset, int length);

        /// <summary>
        /// Gets the bytes at <paramref name="offset"/> as a <see cref="ReadOnlyMemory{T}"/>
        /// that re-checks the source's lifetime on every access.
        /// </summary>
        ReadOnlyMemory<byte> GetMemory(long offset, int length);

        /// <summary>Copies bytes at <paramref name="offset"/> into <paramref name="destination"/>.</summary>
        void CopyTo(long offset, Span<byte> destination);

        /// <summary>Asynchronously copies bytes at <paramref name="offset"/>.</summary>
        ValueTask CopyToAsync(long offset, Memory<byte> destination, CancellationToken cancellationToken);

        /// <summary>Opens a read-only stream over a byte range.</summary>
        Stream OpenStream(long offset, long length);
    }
}
