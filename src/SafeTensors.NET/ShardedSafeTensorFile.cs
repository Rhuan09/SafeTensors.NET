using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Threading;

namespace SafeTensors
{
    /// <summary>
    /// A model split across numbered shard files, presented as one set of tensors.
    /// </summary>
    /// <remarks>
    /// Shards are opened the first time a tensor in them is asked for and stay open until
    /// this object is disposed, so loading three tensors out of a sixty-file checkpoint
    /// maps only the files those tensors live in.
    /// </remarks>
    public sealed class ShardedSafeTensorFile : IDisposable
    {
        private readonly string _baseDirectory;
        private readonly SafeTensorReadOptions _options;

        // Lazy, not a bare factory: ConcurrentDictionary may run a GetOrAdd factory on
        // several threads and keep only one result. With SafeTensorFile as the value that
        // silently leaks memory mappings and file handles for every losing race.
        private readonly ConcurrentDictionary<string, Lazy<SafeTensorFile>> _shards =
            new ConcurrentDictionary<string, Lazy<SafeTensorFile>>(StringComparer.Ordinal);

        private int _disposed;

        /// <summary>Gets the index that maps tensor names to shard files.</summary>
        public ShardIndex Index { get; }

        /// <summary>Gets the directory the shard files are resolved against.</summary>
        public string BaseDirectory => _baseDirectory;

        /// <summary>Gets every tensor name in the model.</summary>
        public IEnumerable<string> Names => Index.WeightMap.Keys;

        /// <summary>Gets the number of tensors in the model.</summary>
        public int Count => Index.WeightMap.Count;

        /// <summary>Gets a tensor by name.</summary>
        /// <exception cref="SafeTensorNotFoundException">No tensor has that name.</exception>
        public TensorView this[string name] => GetTensor(name);

        /// <summary>Creates a sharded model over an index and the directory holding its shards.</summary>
        public ShardedSafeTensorFile(ShardIndex index, string baseDirectory, SafeTensorReadOptions? options = null)
        {
            Index = index ?? throw new ArgumentNullException(nameof(index));

            if (string.IsNullOrEmpty(baseDirectory))
            {
                throw new ArgumentException("Base directory cannot be null or empty.", nameof(baseDirectory));
            }

            _baseDirectory = Path.GetFullPath(baseDirectory);
            _options = options ?? SafeTensorReadOptions.Default;
        }

        /// <summary>
        /// Opens a sharded model from its index file, for example
        /// <c>model.safetensors.index.json</c>.
        /// </summary>
        public static ShardedSafeTensorFile Open(string indexPath, SafeTensorReadOptions? options = null)
        {
            if (string.IsNullOrEmpty(indexPath))
            {
                throw new ArgumentException("Path cannot be null or empty.", nameof(indexPath));
            }

            if (!File.Exists(indexPath))
            {
                throw new FileNotFoundException("Shard index file not found.", indexPath);
            }

            ShardIndex index = ShardIndex.Load(indexPath);
            string directory = Path.GetDirectoryName(Path.GetFullPath(indexPath))!;
            return new ShardedSafeTensorFile(index, directory, options);
        }

        /// <summary>Gets a tensor by name, opening its shard if this is the first request for it.</summary>
        /// <exception cref="SafeTensorNotFoundException">No tensor has that name.</exception>
        public TensorView GetTensor(string name)
        {
            ThrowIfDisposed();

            if (!Index.WeightMap.TryGetValue(name, out string? shard))
            {
                throw new SafeTensorNotFoundException(
                    name,
                    $"Tensor '{name}' is not in the shard index at '{_baseDirectory}'.");
            }

            return OpenShard(shard).GetTensor(name);
        }

        /// <summary>
        /// Tries to get a tensor by name.
        /// </summary>
        /// <remarks>
        /// Returns <c>false</c> only when the name is absent. A shard that is missing from
        /// disk, unreadable or corrupt still throws: swallowing that would report a broken
        /// download as a missing tensor.
        /// </remarks>
        public bool TryGetTensor(string name, out TensorView? tensor)
        {
            ThrowIfDisposed();

            if (!Index.WeightMap.TryGetValue(name, out string? shard))
            {
                tensor = null;
                return false;
            }

            return OpenShard(shard).TryGetTensor(name, out tensor);
        }

        /// <summary>Gets a value indicating whether the index names this tensor.</summary>
        public bool Contains(string name) => Index.WeightMap.ContainsKey(name);

        /// <summary>
        /// Opens a shard by its file name as it appears in the index, or returns the
        /// already-open instance.
        /// </summary>
        public SafeTensorFile OpenShard(string shardFileName)
        {
            ThrowIfDisposed();

            // netstandard2.0 has no state-passing GetOrAdd overload, and the closure costs
            // nothing next to opening a file.
            Lazy<SafeTensorFile> lazy = _shards.GetOrAdd(
                shardFileName,
                name => new Lazy<SafeTensorFile>(
                    () => SafeTensorFile.Open(ResolveShardPath(name), _options),
                    LazyThreadSafetyMode.ExecutionAndPublication));

            SafeTensorFile file = lazy.Value;

            // Losing a race with Dispose would otherwise leave this mapping open forever.
            if (Volatile.Read(ref _disposed) != 0)
            {
                file.Dispose();
                throw new ObjectDisposedException(nameof(ShardedSafeTensorFile));
            }

            return file;
        }

        /// <summary>
        /// Turns a shard name from the index into a full path, refusing anything that
        /// escapes the model directory.
        /// </summary>
        /// <remarks>
        /// The index is untrusted input — it usually arrives with a download from a model
        /// hub — and a <c>weight_map</c> entry is a file name chosen by whoever published
        /// the model. Without this check, an entry of <c>../../../../etc/shadow</c> or an
        /// absolute path would make the loader open a file the caller never named.
        /// </remarks>
        private string ResolveShardPath(string shardFileName)
        {
            if (string.IsNullOrWhiteSpace(shardFileName))
            {
                throw new SafeTensorValidationException("The shard index names an empty shard file.");
            }

            if (Path.IsPathRooted(shardFileName) || shardFileName.IndexOf(':') >= 0)
            {
                throw new SafeTensorValidationException(
                    $"Shard '{shardFileName}' is an absolute path. Shard names must be relative to the index file.");
            }

            string combined = Path.GetFullPath(Path.Combine(_baseDirectory, shardFileName));
            string root = _baseDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
                          + Path.DirectorySeparatorChar;

            if (!combined.StartsWith(root, StringComparison.Ordinal)
                && !string.Equals(combined, _baseDirectory, StringComparison.Ordinal))
            {
                throw new SafeTensorValidationException(
                    $"Shard '{shardFileName}' resolves to '{combined}', outside the model directory " +
                    $"'{_baseDirectory}'. Shard names must not escape it.");
            }

            if (!File.Exists(combined))
            {
                throw new FileNotFoundException(
                    $"Shard '{shardFileName}' named by the index is missing from the model directory.",
                    combined);
            }

            return combined;
        }

        private void ThrowIfDisposed()
        {
            if (Volatile.Read(ref _disposed) != 0)
            {
                throw new ObjectDisposedException(nameof(ShardedSafeTensorFile));
            }
        }

        /// <summary>Closes every shard opened so far.</summary>
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
            {
                return;
            }

            foreach (KeyValuePair<string, Lazy<SafeTensorFile>> entry in _shards)
            {
                if (!entry.Value.IsValueCreated)
                {
                    continue;
                }

                try
                {
                    entry.Value.Value.Dispose();
                }
                catch (SafeTensorException)
                {
                    // A shard that failed to open has nothing to release; the original
                    // failure was already reported to whoever asked for it.
                }
                catch (IOException)
                {
                }
            }

            _shards.Clear();
        }
    }
}
