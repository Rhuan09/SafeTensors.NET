using System;
using System.IO;
using System.Runtime.CompilerServices;

namespace SafeTensors.Internal
{
    /// <summary>
    /// Endian-correct primitives and stream helpers shared by the reader and the writer.
    /// </summary>
    internal static class BinaryUtils
    {
        /// <summary>
        /// Reads a little-endian <see cref="ulong"/>. The SafeTensors header length is
        /// always little-endian regardless of host byte order, so this never uses a raw
        /// reinterpreting load.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong ReadUInt64LittleEndian(ReadOnlySpan<byte> source)
        {
            if (source.Length < 8)
            {
                throw new ArgumentException("Source must be at least 8 bytes.", nameof(source));
            }

            return source[0]
                 | ((ulong)source[1] << 8)
                 | ((ulong)source[2] << 16)
                 | ((ulong)source[3] << 24)
                 | ((ulong)source[4] << 32)
                 | ((ulong)source[5] << 40)
                 | ((ulong)source[6] << 48)
                 | ((ulong)source[7] << 56);
        }

        /// <summary>
        /// Writes a little-endian <see cref="ulong"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteUInt64LittleEndian(Span<byte> destination, ulong value)
        {
            if (destination.Length < 8)
            {
                throw new ArgumentException("Destination must be at least 8 bytes.", nameof(destination));
            }

            destination[0] = (byte)value;
            destination[1] = (byte)(value >> 8);
            destination[2] = (byte)(value >> 16);
            destination[3] = (byte)(value >> 24);
            destination[4] = (byte)(value >> 32);
            destination[5] = (byte)(value >> 40);
            destination[6] = (byte)(value >> 48);
            destination[7] = (byte)(value >> 56);
        }

        /// <summary>
        /// Reads exactly <paramref name="count"/> bytes or throws. Streams are permitted to
        /// return short reads, so a single Read call is never enough.
        /// </summary>
        public static void ReadExactly(Stream stream, byte[] buffer, int offset, int count, string what)
        {
            int total = 0;
            while (total < count)
            {
                int read = stream.Read(buffer, offset + total, count - total);
                if (read == 0)
                {
                    throw new SafeTensorCorruptHeaderException(
                        $"Unexpected end of stream while reading {what}: got {total} of {count} bytes.");
                }

                total += read;
            }
        }

        /// <summary>
        /// Reads exactly <paramref name="destination"/>.Length bytes or throws.
        /// </summary>
        public static void ReadExactly(Stream stream, Span<byte> destination, string what)
        {
#if NETSTANDARD2_0
            byte[] rented = System.Buffers.ArrayPool<byte>.Shared.Rent(destination.Length);
            try
            {
                ReadExactly(stream, rented, 0, destination.Length, what);
                new ReadOnlySpan<byte>(rented, 0, destination.Length).CopyTo(destination);
            }
            finally
            {
                System.Buffers.ArrayPool<byte>.Shared.Return(rented);
            }
#else
            int total = 0;
            while (total < destination.Length)
            {
                int read = stream.Read(destination.Slice(total));
                if (read == 0)
                {
                    throw new SafeTensorCorruptHeaderException(
                        $"Unexpected end of stream while reading {what}: got {total} of {destination.Length} bytes.");
                }

                total += read;
            }
#endif
        }
    }
}
