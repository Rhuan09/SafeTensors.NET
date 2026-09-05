using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Threading;
using System.Threading.Tasks;
using SafeTensors.Internal;

namespace SafeTensors
{
    /// <summary>
    /// An open SafeTensors file: its header, and a view onto every tensor in it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Open a file with <see cref="Open(string, SafeTensorReadOptions?)"/> and it is memory
    /// mapped read-only, so the cost of opening is the header parse and nothing more. The
    /// mapping is taken with <see cref="FileShare.ReadWrite"/> and
    /// <see cref="FileAccess.Read"/>, which is what lets it open files marked read-only,
    /// files on a network share, and files another process already has open.
    /// </para>
    /// <para>
    /// Instances are safe for concurrent readers. Disposing releases the mapping; see
    /// <see cref="TensorView"/> for what that means for spans you are still holding.
    /// </para>
    /// </remarks>
    public sealed class SafeTensorFile : IDisposable
    {
        private readonly ITensorDataSource _source;
        private readonly Dictionary<string, TensorView> _tensors;
        private int _disposed;

        /// <summary>Gets the parsed header.</summary>
        public SafeTensorHeader Header { get; }

        /// <summary>Gets the path this file was opened from, or <c>null</c> for a buffer or stream.</summary>
        public string? FilePath { get; }

        /// <summary>Gets the <c>__metadata__</c> entries.</summary>
        public IReadOnlyDictionary<string, string> Metadata => Header.Metadata;

        /// <summary>Gets every tensor, keyed by name.</summary>
        public IReadOnlyDictionary<string, TensorView> Tensors { get; }

        /// <summary>Gets the tensor names.</summary>
        public IEnumerable<string> Names => _tensors.Keys;

        /// <summary>Gets the number of tensors.</summary>
        public int Count => _tensors.Count;

        /// <summary>Gets a tensor by name.</summary>
        /// <exception cref="SafeTensorNotFoundException">No tensor has that name.</exception>
        public TensorView this[string name] => GetTensor(name);

        internal SafeTensorFile(SafeTensorHeader header, ITensorDataSource source, string? filePath)
        {
            Header = header ?? throw new ArgumentNullException(nameof(header));
            _source = source ?? throw new ArgumentNullException(nameof(source));
            FilePath = filePath;

            _tensors = new Dictionary<string, TensorView>(header.Tensors.Count, StringComparer.Ordinal);
            foreach (KeyValuePair<string, TensorMetadata> pair in header.Tensors)
            {
                _tensors[pair.Key] = new TensorView(pair.Value, source);
            }

            Tensors = new ReadOnlyDictionary<string, TensorView>(_tensors);
        }

        /// <summary>
        /// Opens a file by memory mapping it read-only.
        /// </summary>
        /// <param name="path">Path to a <c>.safetensors</c> file.</param>
        /// <param name="options">Strictness settings; defaults are strict.</param>
        public static SafeTensorFile Open(string path, SafeTensorReadOptions? options = null)
        {
            if (string.IsNullOrEmpty(path))
            {
                throw new ArgumentException("Path cannot be null or empty.", nameof(path));
            }

            options ??= SafeTensorReadOptions.Default;

            var info = new FileInfo(path);
            if (!info.Exists)
            {
                throw new FileNotFoundException("SafeTensors file not found.", path);
            }

            long fileSize = info.Length;
            if (fileSize < 8)
            {
                throw new SafeTensorCorruptHeaderException(
                    $"'{path}' is {fileSize} bytes; a SafeTensors file needs at least the 8-byte header length.");
            }

            // FileAccess.Read plus FileShare.ReadWrite: read-only files open, and another
            // process writing elsewhere in the directory does not block us.
            var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            MemoryMappedFile? mapping = null;
            MemoryMappedViewAccessor? accessor = null;

            try
            {
                SafeTensorHeader header = ReadHeaderCore(stream, fileSize, options);

                mapping = MemoryMappedFile.CreateFromFile(
                    stream,
                    mapName: null,
                    capacity: 0,
                    MemoryMappedFileAccess.Read,
                    HandleInheritability.None,
                    leaveOpen: true);

                accessor = mapping.CreateViewAccessor(0, fileSize, MemoryMappedFileAccess.Read);

                var source = new MemoryMappedDataSource(
                    stream,
                    mapping,
                    accessor,
                    header.DataOffset,
                    fileSize - header.DataOffset);

                return new SafeTensorFile(header, source, path);
            }
            catch
            {
                accessor?.Dispose();
                mapping?.Dispose();
                stream.Dispose();
                throw;
            }
        }

        /// <summary>
        /// Reads a file that is already in memory. Nothing is copied: tensor views point
        /// into <paramref name="buffer"/>.
        /// </summary>
        public static SafeTensorFile Read(ReadOnlyMemory<byte> buffer, SafeTensorReadOptions? options = null)
        {
            options ??= SafeTensorReadOptions.Default;

            if (buffer.Length < 8)
            {
                throw new SafeTensorCorruptHeaderException(
                    $"Buffer is {buffer.Length} bytes; a SafeTensors file needs at least the 8-byte header length.");
            }

            long headerSize = SafeTensorHeader.ValidateHeaderLength(
                BinaryUtils.ReadUInt64LittleEndian(buffer.Span.Slice(0, 8)),
                options);

            if (headerSize + 8 > buffer.Length)
            {
                throw new SafeTensorCorruptHeaderException(
                    $"Header length ({headerSize} + 8 bytes) exceeds the {buffer.Length}-byte buffer.");
            }

            SafeTensorHeader header = SafeTensorHeader.Parse(
                buffer.Slice(8, (int)headerSize),
                headerSize,
                buffer.Length,
                options);

            var source = new MemoryDataSource(buffer, header.DataOffset, buffer.Length - header.DataOffset);
            return new SafeTensorFile(header, source, filePath: null);
        }

        /// <summary>Reads a file that is already in memory.</summary>
        public static SafeTensorFile Read(byte[] buffer, SafeTensorReadOptions? options = null)
        {
            if (buffer is null)
            {
                throw new ArgumentNullException(nameof(buffer));
            }

            return Read(new ReadOnlyMemory<byte>(buffer), options);
        }

        /// <summary>
        /// Reads from a seekable stream, fetching tensor bytes on demand rather than loading
        /// the file.
        /// </summary>
        /// <param name="stream">A seekable stream positioned at the start of the file.</param>
        /// <param name="leaveOpen">Keep <paramref name="stream"/> open when this file is disposed.</param>
        /// <param name="options">Strictness settings; defaults are strict.</param>
        public static SafeTensorFile Read(
            Stream stream,
            bool leaveOpen = false,
            SafeTensorReadOptions? options = null)
        {
            if (stream is null)
            {
                throw new ArgumentNullException(nameof(stream));
            }

            if (!stream.CanSeek)
            {
                throw new ArgumentException(
                    "Random tensor access needs a seekable stream. Copy a forward-only stream " +
                    "into a MemoryStream, or a file, first.",
                    nameof(stream));
            }

            options ??= SafeTensorReadOptions.Default;

            try
            {
                long available = stream.Length - stream.Position;
                SafeTensorHeader header = ReadHeaderCore(stream, available, options);
                long dataOffset = stream.Position;

                var source = new StreamDataSource(stream, dataOffset, stream.Length - dataOffset, leaveOpen);
                return new SafeTensorFile(header, source, filePath: null);
            }
            catch
            {
                if (!leaveOpen)
                {
                    stream.Dispose();
                }

                throw;
            }
        }

        /// <summary>
        /// Reads only the header, without mapping the file.
        /// </summary>
        /// <remarks>
        /// This is how you inspect a multi-gigabyte checkpoint for the cost of one seek and
        /// one small read: names, shapes, dtypes and sizes all live in the header.
        /// </remarks>
        public static SafeTensorHeader ReadHeader(string path, SafeTensorReadOptions? options = null)
        {
            if (string.IsNullOrEmpty(path))
            {
                throw new ArgumentException("Path cannot be null or empty.", nameof(path));
            }

            var info = new FileInfo(path);
            if (!info.Exists)
            {
                throw new FileNotFoundException("SafeTensors file not found.", path);
            }

            if (info.Length < 8)
            {
                throw new SafeTensorCorruptHeaderException(
                    $"'{path}' is {info.Length} bytes; a SafeTensors file needs at least the 8-byte header length.");
            }

            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            return ReadHeaderCore(stream, info.Length, options ?? SafeTensorReadOptions.Default);
        }

        /// <summary>Reads only the header from a stream.</summary>
        public static SafeTensorHeader ReadHeader(
            Stream stream,
            bool leaveOpen = true,
            SafeTensorReadOptions? options = null)
        {
            if (stream is null)
            {
                throw new ArgumentNullException(nameof(stream));
            }

            try
            {
                long available = stream.CanSeek ? stream.Length - stream.Position : -1;
                return ReadHeaderCore(stream, available, options ?? SafeTensorReadOptions.Default);
            }
            finally
            {
                if (!leaveOpen)
                {
                    stream.Dispose();
                }
            }
        }

        /// <summary>Asynchronously reads only the header from a stream.</summary>
        public static async Task<SafeTensorHeader> ReadHeaderAsync(
            Stream stream,
            bool leaveOpen = true,
            SafeTensorReadOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            if (stream is null)
            {
                throw new ArgumentNullException(nameof(stream));
            }

            options ??= SafeTensorReadOptions.Default;

            try
            {
                byte[] prefix = new byte[8];
                await ReadExactlyAsync(stream, prefix, 8, "the 8-byte header length", cancellationToken)
                    .ConfigureAwait(false);

                long headerSize = SafeTensorHeader.ValidateHeaderLength(
                    BinaryUtils.ReadUInt64LittleEndian(prefix),
                    options);

                byte[] json = new byte[headerSize];
                await ReadExactlyAsync(stream, json, (int)headerSize, "the JSON header", cancellationToken)
                    .ConfigureAwait(false);

                long available = stream.CanSeek ? stream.Length : -1;
                return SafeTensorHeader.Parse(json, headerSize, available, options);
            }
            finally
            {
                if (!leaveOpen)
                {
                    stream.Dispose();
                }
            }
        }

        private static SafeTensorHeader ReadHeaderCore(Stream stream, long totalSize, SafeTensorReadOptions options)
        {
            long headerSize = SafeTensorHeader.ReadHeaderLength(stream, options);

            if (totalSize >= 0 && headerSize + 8 > totalSize)
            {
                throw new SafeTensorCorruptHeaderException(
                    $"Header length ({headerSize} + 8 bytes) exceeds the total size ({totalSize} bytes).");
            }

            byte[] json = new byte[headerSize];
            BinaryUtils.ReadExactly(stream, json, 0, (int)headerSize, "the JSON header");
            return SafeTensorHeader.Parse(json, headerSize, totalSize, options);
        }

        private static async Task ReadExactlyAsync(
            Stream stream,
            byte[] buffer,
            int count,
            string what,
            CancellationToken cancellationToken)
        {
            int total = 0;
            while (total < count)
            {
                int read = await stream.ReadAsync(buffer, total, count - total, cancellationToken)
                    .ConfigureAwait(false);
                if (read == 0)
                {
                    throw new SafeTensorCorruptHeaderException(
                        $"Unexpected end of stream while reading {what}: got {total} of {count} bytes.");
                }

                total += read;
            }
        }

        /// <summary>Gets a tensor by name.</summary>
        /// <exception cref="SafeTensorNotFoundException">No tensor has that name.</exception>
        public TensorView GetTensor(string name)
        {
            ThrowIfDisposed();

            if (_tensors.TryGetValue(name, out TensorView? view))
            {
                return view;
            }

            throw new SafeTensorNotFoundException(
                name,
                FilePath is null
                    ? $"Tensor '{name}' is not in this SafeTensors file."
                    : $"Tensor '{name}' is not in '{FilePath}'.");
        }

        /// <summary>Tries to get a tensor by name.</summary>
        public bool TryGetTensor(string name, out TensorView? tensor)
        {
            ThrowIfDisposed();
            return _tensors.TryGetValue(name, out tensor);
        }

        /// <summary>Gets a value indicating whether a tensor with this name exists.</summary>
        public bool Contains(string name)
        {
            ThrowIfDisposed();
            return _tensors.ContainsKey(name);
        }

        private void ThrowIfDisposed()
        {
            if (Volatile.Read(ref _disposed) != 0)
            {
                throw new ObjectDisposedException(nameof(SafeTensorFile));
            }
        }

        /// <summary>
        /// Releases the mapping or stream backing this file.
        /// </summary>
        /// <remarks>
        /// Spans previously handed out by <see cref="TensorView.AsSpan{T}()"/> over a memory
        /// mapping become invalid here. Memory obtained from
        /// <see cref="TensorView.AsMemory{T}()"/> throws <see cref="ObjectDisposedException"/>
        /// instead, and arrays from <see cref="TensorView.ToArray{T}()"/> are unaffected.
        /// </remarks>
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
            {
                _source.Dispose();
            }
        }
    }
}
