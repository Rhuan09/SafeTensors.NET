namespace SafeTensors
{
    /// <summary>
    /// Limits and strictness knobs applied while parsing a SafeTensors header.
    /// </summary>
    /// <remarks>
    /// The defaults are the strict ones. A checkpoint is an untrusted input — it usually
    /// arrives over the network from a model hub — so the reader validates its internal
    /// consistency before handing out a single byte, and you opt out of that deliberately
    /// rather than opting in.
    /// </remarks>
    public sealed class SafeTensorReadOptions
    {
        /// <summary>
        /// The options used when none are supplied.
        /// </summary>
        public static SafeTensorReadOptions Default { get; } = new SafeTensorReadOptions();

        /// <summary>
        /// Largest header this reader will allocate for, in bytes. Default 100 MiB.
        /// </summary>
        /// <remarks>
        /// The header length is an unprefixed 64-bit integer at offset 0, so a hostile file
        /// can ask for 16 exabytes before a single byte of it has been validated. This cap
        /// is what turns that into an exception instead of an allocation. Real headers, even
        /// for models with hundreds of thousands of tensors, are a few megabytes.
        /// </remarks>
        public long MaxHeaderSize { get; init; } = 100L * 1024 * 1024;

        /// <summary>
        /// Allow gaps between tensor byte ranges. Default <c>false</c>.
        /// </summary>
        /// <remarks>
        /// Overlaps are rejected regardless of this setting: two tensors that alias the same
        /// bytes are never valid. This flag only controls whether unclaimed padding between
        /// ranges is tolerated.
        /// </remarks>
        public bool AllowNonContiguousData { get; init; }

        /// <summary>
        /// Allow bytes after the last tensor. Default <c>true</c>.
        /// </summary>
        /// <remarks>
        /// Some producers pad files to a block boundary. Trailing slack is harmless because
        /// nothing addresses it, so it is accepted by default.
        /// </remarks>
        public bool AllowTrailingBytes { get; init; } = true;
    }
}
