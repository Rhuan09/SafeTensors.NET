using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace SafeTensors.Native
{
    /// <summary>
    /// C-layout description of one tensor, filled in by <c>safetensors_get_tensor_info</c>.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct SafeTensorNativeInfo
    {
        /// <summary>The dtype, as the numeric value of <see cref="SafeTensorDType"/>.</summary>
        public int DType;

        /// <summary>Number of dimensions.</summary>
        public int Rank;

        /// <summary>Pointer to <see cref="Rank"/> dimensions. Owned by the handle.</summary>
        public long* Shape;

        /// <summary>Tensor length in bytes.</summary>
        public ulong ByteLength;

        /// <summary>Number of elements.</summary>
        public ulong ElementCount;
    }

    /// <summary>
    /// The C ABI. Compiled with Native AOT into a shared library that C, C++, Rust, Python
    /// or anything else with an FFI can load.
    /// </summary>
    /// <remarks>
    /// <para><b>Ownership.</b> Two functions return heap memory the caller must release with
    /// <c>safetensors_free_string</c>: <c>safetensors_get_last_error</c> and
    /// <c>safetensors_get_metadata</c>. Everything else returns either a borrowed pointer
    /// valid until <c>safetensors_close</c>, or a status code.
    /// </para>
    /// <para><b>Errors.</b> Functions returning a pointer signal failure with <c>NULL</c>;
    /// functions returning <c>int</c> signal it with a negative value. Either way the
    /// detail is in <c>safetensors_get_last_error</c>, which is per-thread.
    /// </para>
    /// </remarks>
    public static unsafe class NativeExports
    {
        [ThreadStatic]
        private static string? s_lastError;

        private sealed class NativeContext : IDisposable
        {
            private readonly object _lock = new();
            private readonly Dictionary<string, GCHandle> _pinnedShapes = new(StringComparer.Ordinal);

            public SafeTensorFile File { get; }

            public List<string> TensorNames { get; }

            public NativeContext(SafeTensorFile file)
            {
                File = file;
                TensorNames = new List<string>(file.Names);
            }

            /// <summary>
            /// Pins a copy of the tensor's shape so the caller gets a pointer that stays
            /// valid for the life of the handle.
            /// </summary>
            /// <remarks>
            /// The alternative is making the caller supply a buffer, which doubles the number
            /// of round trips for something that is at most a few dozen bytes per tensor.
            /// The pins are released by <c>safetensors_close</c>.
            /// </remarks>
            public long* PinShape(string name, long[] shape)
            {
                lock (_lock)
                {
                    if (!_pinnedShapes.TryGetValue(name, out GCHandle handle))
                    {
                        handle = GCHandle.Alloc(shape, GCHandleType.Pinned);
                        _pinnedShapes[name] = handle;
                    }

                    return (long*)handle.AddrOfPinnedObject();
                }
            }

            public void Dispose()
            {
                lock (_lock)
                {
                    foreach (GCHandle handle in _pinnedShapes.Values)
                    {
                        if (handle.IsAllocated)
                        {
                            handle.Free();
                        }
                    }

                    _pinnedShapes.Clear();
                }

                File.Dispose();
            }
        }

        private static void SetError(string message) => s_lastError = message;

        private static void SetError(Exception exception) => s_lastError = exception.Message;

        private static void ClearError() => s_lastError = null;

        private static bool TryGetContext(void* handle, out NativeContext context)
        {
            if (handle != null && GCHandle.FromIntPtr((IntPtr)handle).Target is NativeContext found)
            {
                context = found;
                return true;
            }

            SetError("Invalid or already-closed SafeTensors handle.");
            context = null!;
            return false;
        }

        private static byte* AllocUtf8(string value)
        {
            int byteCount = Encoding.UTF8.GetByteCount(value);
            byte* buffer = (byte*)NativeMemory.Alloc((nuint)byteCount + 1);

            fixed (char* chars = value)
            {
                Encoding.UTF8.GetBytes(chars, value.Length, buffer, byteCount);
            }

            buffer[byteCount] = 0;
            return buffer;
        }

        /// <summary>
        /// Returns the last error on this thread as a NUL-terminated UTF-8 string, or NULL.
        /// The caller owns the result and must pass it to <c>safetensors_free_string</c>.
        /// </summary>
        [UnmanagedCallersOnly(EntryPoint = "safetensors_get_last_error")]
        public static byte* GetLastError()
        {
            string? error = s_lastError;
            return string.IsNullOrEmpty(error) ? null : AllocUtf8(error!);
        }

        /// <summary>Frees a string returned by this library.</summary>
        [UnmanagedCallersOnly(EntryPoint = "safetensors_free_string")]
        public static void FreeString(byte* value)
        {
            if (value != null)
            {
                NativeMemory.Free(value);
            }
        }

        /// <summary>Opens a file. Returns NULL on failure.</summary>
        [UnmanagedCallersOnly(EntryPoint = "safetensors_open")]
        public static void* Open(byte* utf8Path)
        {
            ClearError();

            if (utf8Path == null)
            {
                SetError("Path pointer is NULL.");
                return null;
            }

            try
            {
                string path = Marshal.PtrToStringUTF8((IntPtr)utf8Path) ?? string.Empty;
                var context = new NativeContext(SafeTensorFile.Open(path));
                return (void*)GCHandle.ToIntPtr(GCHandle.Alloc(context));
            }
            catch (Exception ex)
            {
                SetError(ex);
                return null;
            }
        }

        /// <summary>
        /// Closes a handle, releasing the mapping. Every pointer obtained from it becomes
        /// invalid.
        /// </summary>
        [UnmanagedCallersOnly(EntryPoint = "safetensors_close")]
        public static void Close(void* handle)
        {
            if (handle == null)
            {
                return;
            }

            try
            {
                GCHandle gcHandle = GCHandle.FromIntPtr((IntPtr)handle);
                (gcHandle.Target as NativeContext)?.Dispose();
                gcHandle.Free();
            }
            catch (Exception ex)
            {
                SetError(ex);
            }
        }

        /// <summary>Returns the number of tensors, or -1 on failure.</summary>
        [UnmanagedCallersOnly(EntryPoint = "safetensors_get_tensor_count")]
        public static int GetTensorCount(void* handle)
        {
            ClearError();

            try
            {
                return TryGetContext(handle, out NativeContext context) ? context.File.Count : -1;
            }
            catch (Exception ex)
            {
                SetError(ex);
                return -1;
            }
        }

        /// <summary>
        /// Copies the tensor name at <paramref name="index"/> into
        /// <paramref name="buffer"/> as NUL-terminated UTF-8.
        /// </summary>
        /// <returns>
        /// The name length in bytes excluding the terminator, or -1 on failure. When the
        /// buffer is too small the required size including the terminator is returned as a
        /// negative number, so <c>-n</c> means "call again with n bytes".
        /// </returns>
        [UnmanagedCallersOnly(EntryPoint = "safetensors_get_tensor_name")]
        public static int GetTensorName(void* handle, int index, byte* buffer, int bufferSize)
        {
            ClearError();

            if (buffer == null || bufferSize <= 0)
            {
                SetError("Output buffer is NULL or empty.");
                return -1;
            }

            try
            {
                if (!TryGetContext(handle, out NativeContext context))
                {
                    return -1;
                }

                if (index < 0 || index >= context.TensorNames.Count)
                {
                    SetError($"Index {index} is out of range; the file has {context.TensorNames.Count} tensors.");
                    return -1;
                }

                string name = context.TensorNames[index];
                int byteCount = Encoding.UTF8.GetByteCount(name);

                if (byteCount + 1 > bufferSize)
                {
                    SetError($"Buffer of {bufferSize} bytes is too small for a {byteCount + 1}-byte name.");
                    return -(byteCount + 1);
                }

                fixed (char* chars = name)
                {
                    Encoding.UTF8.GetBytes(chars, name.Length, buffer, byteCount);
                }

                buffer[byteCount] = 0;
                return byteCount;
            }
            catch (Exception ex)
            {
                SetError(ex);
                return -1;
            }
        }

        /// <summary>Fills <paramref name="info"/> for a tensor. Returns 0, or -1 on failure.</summary>
        [UnmanagedCallersOnly(EntryPoint = "safetensors_get_tensor_info")]
        public static int GetTensorInfo(void* handle, byte* utf8Name, SafeTensorNativeInfo* info)
        {
            ClearError();

            if (utf8Name == null || info == null)
            {
                SetError("Name or output pointer is NULL.");
                return -1;
            }

            try
            {
                if (!TryGetContext(handle, out NativeContext context))
                {
                    return -1;
                }

                string name = Marshal.PtrToStringUTF8((IntPtr)utf8Name) ?? string.Empty;
                if (!context.File.TryGetTensor(name, out TensorView? tensor) || tensor is null)
                {
                    SetError($"Tensor '{name}' was not found.");
                    return -1;
                }

                long[] shape = tensor.Metadata.ToShapeArray();

                info->DType = (int)tensor.DType;
                info->Rank = shape.Length;
                info->Shape = shape.Length == 0 ? null : context.PinShape(name, shape);
                info->ByteLength = (ulong)tensor.ByteLength;
                info->ElementCount = (ulong)tensor.ElementCount;

                return 0;
            }
            catch (Exception ex)
            {
                SetError(ex);
                return -1;
            }
        }

        /// <summary>
        /// Returns a pointer straight into the mapped file, valid until
        /// <c>safetensors_close</c>. Returns NULL on failure.
        /// </summary>
        /// <remarks>
        /// This is the whole point of the native layer: a consumer in C or Rust gets the
        /// weights at their address in the page cache, with no copy and no marshalling.
        /// </remarks>
        [UnmanagedCallersOnly(EntryPoint = "safetensors_get_tensor_data_ptr")]
        public static void* GetTensorDataPtr(void* handle, byte* utf8Name, ulong* byteLength)
        {
            ClearError();

            if (utf8Name == null)
            {
                SetError("Name pointer is NULL.");
                return null;
            }

            try
            {
                if (!TryGetContext(handle, out NativeContext context))
                {
                    return null;
                }

                string name = Marshal.PtrToStringUTF8((IntPtr)utf8Name) ?? string.Empty;
                if (!context.File.TryGetTensor(name, out TensorView? tensor) || tensor is null)
                {
                    SetError($"Tensor '{name}' was not found.");
                    return null;
                }

                void* pointer = tensor.DangerousGetPointer();
                if (pointer == null)
                {
                    SetError($"Tensor '{name}' is not backed by addressable memory. Use safetensors_copy_tensor_data.");
                    return null;
                }

                if (byteLength != null)
                {
                    *byteLength = (ulong)tensor.ByteLength;
                }

                return pointer;
            }
            catch (Exception ex)
            {
                SetError(ex);
                return null;
            }
        }

        /// <summary>Copies a tensor into a caller-owned buffer. Returns 0, or -1 on failure.</summary>
        [UnmanagedCallersOnly(EntryPoint = "safetensors_copy_tensor_data")]
        public static int CopyTensorData(void* handle, byte* utf8Name, void* destination, ulong destinationSize)
        {
            ClearError();

            if (utf8Name == null || destination == null)
            {
                SetError("Name or destination pointer is NULL.");
                return -1;
            }

            try
            {
                if (!TryGetContext(handle, out NativeContext context))
                {
                    return -1;
                }

                string name = Marshal.PtrToStringUTF8((IntPtr)utf8Name) ?? string.Empty;
                if (!context.File.TryGetTensor(name, out TensorView? tensor) || tensor is null)
                {
                    SetError($"Tensor '{name}' was not found.");
                    return -1;
                }

                if ((ulong)tensor.ByteLength > destinationSize)
                {
                    SetError($"Destination of {destinationSize} bytes is too small for a {tensor.ByteLength}-byte tensor.");
                    return -1;
                }

                if (tensor.ByteLength > int.MaxValue)
                {
                    SetError($"Tensor '{name}' is {tensor.ByteLength} bytes, more than one copy can address. " +
                             "Use safetensors_get_tensor_data_ptr.");
                    return -1;
                }

                tensor.CopyTo(new Span<byte>(destination, (int)tensor.ByteLength));
                return 0;
            }
            catch (Exception ex)
            {
                SetError(ex);
                return -1;
            }
        }

        /// <summary>
        /// Returns a <c>__metadata__</c> value as NUL-terminated UTF-8, or NULL when absent.
        /// The caller owns the result and must pass it to <c>safetensors_free_string</c>.
        /// </summary>
        [UnmanagedCallersOnly(EntryPoint = "safetensors_get_metadata")]
        public static byte* GetMetadata(void* handle, byte* utf8Key)
        {
            ClearError();

            if (utf8Key == null)
            {
                SetError("Key pointer is NULL.");
                return null;
            }

            try
            {
                if (!TryGetContext(handle, out NativeContext context))
                {
                    return null;
                }

                string key = Marshal.PtrToStringUTF8((IntPtr)utf8Key) ?? string.Empty;
                return context.File.Metadata.TryGetValue(key, out string? value) && value is not null
                    ? AllocUtf8(value)
                    : null;
            }
            catch (Exception ex)
            {
                SetError(ex);
                return null;
            }
        }
    }
}
