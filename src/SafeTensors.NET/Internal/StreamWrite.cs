using System;
using System.Buffers;
using System.IO;

namespace SafeTensors.Internal
{
    /// <summary>
    /// Writes spans to a stream on frameworks whose <see cref="Stream"/> only accepts arrays.
    /// </summary>
    internal static class StreamWrite
    {
        private const int ChunkSize = 128 * 1024;

        public static void Span(Stream destination, ReadOnlySpan<byte> source)
        {
#if NETSTANDARD2_0
            // Chunked through a pooled buffer rather than source.ToArray(): a tensor can be
            // gigabytes, and the point of not copying it into the builder is lost if the
            // writer copies it all here instead.
            byte[] buffer = ArrayPool<byte>.Shared.Rent(Math.Min(ChunkSize, Math.Max(source.Length, 1)));
            try
            {
                int written = 0;
                while (written < source.Length)
                {
                    int take = Math.Min(buffer.Length, source.Length - written);
                    source.Slice(written, take).CopyTo(buffer);
                    destination.Write(buffer, 0, take);
                    written += take;
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(buffer);
            }
#else
            destination.Write(source);
#endif
        }
    }
}
