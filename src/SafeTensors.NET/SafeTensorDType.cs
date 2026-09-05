using System;

namespace SafeTensors
{
    /// <summary>
    /// Element types defined by the SafeTensors format.
    /// </summary>
    /// <remarks>
    /// The names match the strings written into the header verbatim, so
    /// <c>SafeTensorDType.BF16</c> is the <c>"BF16"</c> a Python or Rust producer writes.
    /// </remarks>
    public enum SafeTensorDType
    {
        /// <summary>One byte per element, 0 or 1.</summary>
        BOOL,

        /// <summary>Unsigned 8-bit integer.</summary>
        U8,

        /// <summary>Signed 8-bit integer.</summary>
        I8,

        /// <summary>Signed 16-bit integer.</summary>
        I16,

        /// <summary>Unsigned 16-bit integer.</summary>
        U16,

        /// <summary>IEEE 754 binary16 half precision.</summary>
        F16,

        /// <summary>Brain floating point: 8 exponent bits, 7 significand bits.</summary>
        BF16,

        /// <summary>Signed 32-bit integer.</summary>
        I32,

        /// <summary>Unsigned 32-bit integer.</summary>
        U32,

        /// <summary>IEEE 754 binary32 single precision.</summary>
        F32,

        /// <summary>IEEE 754 binary64 double precision.</summary>
        F64,

        /// <summary>Signed 64-bit integer.</summary>
        I64,

        /// <summary>Unsigned 64-bit integer.</summary>
        U64,

        /// <summary>8-bit float, 4 exponent bits and 3 significand bits.</summary>
        F8_E4M3,

        /// <summary>8-bit float, 5 exponent bits and 2 significand bits.</summary>
        F8_E5M2
    }

    /// <summary>
    /// Size, naming and CLR mapping for <see cref="SafeTensorDType"/>.
    /// </summary>
    /// <remarks>
    /// These are plain static methods rather than extension methods because most of them
    /// are factories that have no receiver to extend. The two that read naturally as
    /// extensions are also exposed that way on <see cref="SafeTensorDTypeExtensions"/>.
    /// </remarks>
    public static class DTypes
    {
        /// <summary>
        /// Gets the size of one element in bits.
        /// </summary>
        /// <remarks>
        /// Sizes are expressed in bits rather than bytes so that sub-byte element types can
        /// be added later without changing this signature. Every type defined today is a
        /// whole number of bytes.
        /// </remarks>
        public static int BitSize(SafeTensorDType dtype) => dtype switch
        {
            SafeTensorDType.BOOL => 8,
            SafeTensorDType.U8 => 8,
            SafeTensorDType.I8 => 8,
            SafeTensorDType.F8_E4M3 => 8,
            SafeTensorDType.F8_E5M2 => 8,
            SafeTensorDType.I16 => 16,
            SafeTensorDType.U16 => 16,
            SafeTensorDType.F16 => 16,
            SafeTensorDType.BF16 => 16,
            SafeTensorDType.I32 => 32,
            SafeTensorDType.U32 => 32,
            SafeTensorDType.F32 => 32,
            SafeTensorDType.I64 => 64,
            SafeTensorDType.U64 => 64,
            SafeTensorDType.F64 => 64,
            _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, "Unknown SafeTensors dtype.")
        };

        /// <summary>
        /// Gets the size of one element in bytes.
        /// </summary>
        public static int ByteSize(SafeTensorDType dtype) => BitSize(dtype) / 8;

        /// <summary>
        /// Gets the number of bytes occupied by <paramref name="elementCount"/> elements.
        /// </summary>
        /// <exception cref="SafeTensorValidationException">The product overflows 64 bits.</exception>
        public static long ByteLength(SafeTensorDType dtype, long elementCount)
        {
            if (elementCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(elementCount));
            }

            try
            {
                return checked(elementCount * ByteSize(dtype));
            }
            catch (OverflowException ex)
            {
                throw new SafeTensorValidationException(
                    $"Element count {elementCount} of dtype {dtype} overflows a 64-bit byte length.", ex);
            }
        }

        /// <summary>
        /// Gets the canonical header string for a dtype, for example <c>"BF16"</c>.
        /// </summary>
        public static string ToHeaderString(SafeTensorDType dtype) => dtype switch
        {
            SafeTensorDType.BOOL => "BOOL",
            SafeTensorDType.U8 => "U8",
            SafeTensorDType.I8 => "I8",
            SafeTensorDType.I16 => "I16",
            SafeTensorDType.U16 => "U16",
            SafeTensorDType.F16 => "F16",
            SafeTensorDType.BF16 => "BF16",
            SafeTensorDType.I32 => "I32",
            SafeTensorDType.U32 => "U32",
            SafeTensorDType.F32 => "F32",
            SafeTensorDType.F64 => "F64",
            SafeTensorDType.I64 => "I64",
            SafeTensorDType.U64 => "U64",
            SafeTensorDType.F8_E4M3 => "F8_E4M3",
            SafeTensorDType.F8_E5M2 => "F8_E5M2",
            _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, "Unknown SafeTensors dtype.")
        };

        /// <summary>
        /// Parses a dtype string from a header. Canonical names are accepted, as are the
        /// common framework aliases such as <c>float32</c>, <c>bfloat16</c> and <c>half</c>.
        /// </summary>
        public static bool TryParse(string? text, out SafeTensorDType dtype)
        {
            if (string.IsNullOrEmpty(text))
            {
                dtype = default;
                return false;
            }

            switch (text!.Trim().ToUpperInvariant())
            {
                case "BOOL": dtype = SafeTensorDType.BOOL; return true;
                case "U8":
                case "UINT8": dtype = SafeTensorDType.U8; return true;
                case "I8":
                case "INT8": dtype = SafeTensorDType.I8; return true;
                case "I16":
                case "INT16": dtype = SafeTensorDType.I16; return true;
                case "U16":
                case "UINT16": dtype = SafeTensorDType.U16; return true;
                case "F16":
                case "FLOAT16":
                case "HALF": dtype = SafeTensorDType.F16; return true;
                case "BF16":
                case "BFLOAT16": dtype = SafeTensorDType.BF16; return true;
                case "I32":
                case "INT32": dtype = SafeTensorDType.I32; return true;
                case "U32":
                case "UINT32": dtype = SafeTensorDType.U32; return true;
                case "F32":
                case "FLOAT32":
                case "FLOAT": dtype = SafeTensorDType.F32; return true;
                case "F64":
                case "FLOAT64":
                case "DOUBLE": dtype = SafeTensorDType.F64; return true;
                case "I64":
                case "INT64": dtype = SafeTensorDType.I64; return true;
                case "U64":
                case "UINT64": dtype = SafeTensorDType.U64; return true;
                case "F8_E4M3": dtype = SafeTensorDType.F8_E4M3; return true;
                case "F8_E5M2": dtype = SafeTensorDType.F8_E5M2; return true;
                default: dtype = default; return false;
            }
        }

        /// <summary>
        /// Parses a dtype string, throwing if it is not recognised.
        /// </summary>
        public static SafeTensorDType Parse(string text)
        {
            if (TryParse(text, out SafeTensorDType dtype))
            {
                return dtype;
            }

            throw new ArgumentException($"Unsupported SafeTensors dtype: {text}", nameof(text));
        }

        /// <summary>
        /// Maps a CLR element type to its dtype.
        /// </summary>
        public static SafeTensorDType FromClrType<T>()
            where T : unmanaged
            => FromClrType(typeof(T));

        /// <summary>
        /// Maps a CLR element type to its dtype.
        /// </summary>
        /// <remarks>
        /// <see cref="byte"/> maps to <see cref="SafeTensorDType.U8"/> and <see cref="bool"/>
        /// to <see cref="SafeTensorDType.BOOL"/>. The 8-bit float types have no CLR
        /// counterpart and must be written through an overload that takes raw bytes and an
        /// explicit dtype.
        /// </remarks>
        public static SafeTensorDType FromClrType(Type type)
        {
            if (type is null)
            {
                throw new ArgumentNullException(nameof(type));
            }

            if (type == typeof(bool)) return SafeTensorDType.BOOL;
            if (type == typeof(byte)) return SafeTensorDType.U8;
            if (type == typeof(sbyte)) return SafeTensorDType.I8;
            if (type == typeof(short)) return SafeTensorDType.I16;
            if (type == typeof(ushort)) return SafeTensorDType.U16;
            if (type == typeof(Float16)) return SafeTensorDType.F16;
#if NET5_0_OR_GREATER
            if (type == typeof(Half)) return SafeTensorDType.F16;
#endif
            if (type == typeof(BFloat16)) return SafeTensorDType.BF16;
            if (type == typeof(int)) return SafeTensorDType.I32;
            if (type == typeof(uint)) return SafeTensorDType.U32;
            if (type == typeof(float)) return SafeTensorDType.F32;
            if (type == typeof(double)) return SafeTensorDType.F64;
            if (type == typeof(long)) return SafeTensorDType.I64;
            if (type == typeof(ulong)) return SafeTensorDType.U64;

            throw new NotSupportedException(
                $"CLR type '{type.FullName}' has no SafeTensors dtype. Write it through an " +
                "AddTensor overload that takes an explicit dtype and raw bytes.");
        }
    }

    /// <summary>
    /// Ergonomic extension forms of the two <see cref="DTypes"/> members that read naturally
    /// with a receiver.
    /// </summary>
    public static class SafeTensorDTypeExtensions
    {
        /// <summary>Gets the size of one element in bytes.</summary>
        public static int GetByteSize(this SafeTensorDType dtype) => DTypes.ByteSize(dtype);

        /// <summary>Gets the canonical header string, for example <c>"BF16"</c>.</summary>
        public static string ToDTypeString(this SafeTensorDType dtype) => DTypes.ToHeaderString(dtype);
    }
}
