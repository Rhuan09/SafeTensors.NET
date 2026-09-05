using System;
using System.Globalization;
using System.Runtime.InteropServices;
using SafeTensors.Internal;

namespace SafeTensors
{
    /// <summary>
    /// Brain floating point: the top 16 bits of a binary32, so 8 exponent bits and 7
    /// significand bits.
    /// </summary>
    /// <remarks>
    /// BF16 trades precision for range, keeping binary32's exponent so that conversion to
    /// <see cref="float"/> is a shift and conversion back is a rounded truncation. It is
    /// the dominant weight format for large models, which is why it gets a first-class type
    /// here rather than being handed back as raw <see cref="ushort"/>.
    /// </remarks>
    [StructLayout(LayoutKind.Sequential, Size = 2)]
    public readonly struct BFloat16 : IEquatable<BFloat16>, IComparable<BFloat16>, IFormattable
    {
        private readonly ushort _bits;

        /// <summary>Creates a value from its raw 16-bit encoding.</summary>
        public BFloat16(ushort rawBits) => _bits = rawBits;

        /// <summary>Gets the raw 16-bit encoding.</summary>
        public ushort RawBits => _bits;

        /// <summary>Gets a value indicating whether this is NaN.</summary>
        public bool IsNaN => (_bits & 0x7F80) == 0x7F80 && (_bits & 0x007F) != 0;

        /// <summary>Gets a value indicating whether this is positive or negative infinity.</summary>
        public bool IsInfinity => (_bits & 0x7FFF) == 0x7F80;

        /// <summary>Converts to <see cref="float"/>. Always exact: BF16 is a truncated binary32.</summary>
        public static implicit operator float(BFloat16 value)
            => FloatBits.ToSingle((uint)value._bits << 16);

        /// <summary>Converts from <see cref="float"/>, rounding to nearest with ties to even.</summary>
        public static explicit operator BFloat16(float value) => new BFloat16(SingleToBits(value));

        /// <summary>Converts to <see cref="double"/>. Always exact.</summary>
        public static implicit operator double(BFloat16 value)
            => FloatBits.ToSingle((uint)value._bits << 16);

        /// <summary>Converts a <see cref="float"/> to raw bfloat16, round to nearest even.</summary>
        internal static ushort SingleToBits(float value)
        {
            uint f = FloatBits.ToUInt32(value);

            // A NaN must stay a NaN. Adding the rounding bias to a significand of all ones
            // would carry into the exponent and produce infinity, so handle it first.
            if ((f & 0x7FFFFFFFu) > 0x7F800000u)
            {
                return (ushort)((f >> 16) | 0x0040u);
            }

            uint lsb = (f >> 16) & 1u;
            uint rounded = f + 0x7FFFu + lsb;
            return (ushort)(rounded >> 16);
        }

        /// <inheritdoc />
        public bool Equals(BFloat16 other) => (float)this == (float)other;

        /// <inheritdoc />
        public override bool Equals(object? obj) => obj is BFloat16 other && Equals(other);

        /// <inheritdoc />
        public override int GetHashCode() => ((float)this).GetHashCode();

        /// <inheritdoc />
        public int CompareTo(BFloat16 other) => ((float)this).CompareTo((float)other);

        /// <inheritdoc />
        public override string ToString() => ((float)this).ToString(CultureInfo.CurrentCulture);

        /// <inheritdoc />
        public string ToString(string? format, IFormatProvider? formatProvider)
            => ((float)this).ToString(format, formatProvider);

        /// <summary>Compares two values numerically. NaN is never equal to anything.</summary>
        public static bool operator ==(BFloat16 left, BFloat16 right) => (float)left == (float)right;

        /// <summary>Compares two values numerically.</summary>
        public static bool operator !=(BFloat16 left, BFloat16 right) => !(left == right);

        /// <summary>Compares two values numerically.</summary>
        public static bool operator <(BFloat16 left, BFloat16 right) => (float)left < (float)right;

        /// <summary>Compares two values numerically.</summary>
        public static bool operator >(BFloat16 left, BFloat16 right) => (float)left > (float)right;

        /// <summary>Compares two values numerically.</summary>
        public static bool operator <=(BFloat16 left, BFloat16 right) => (float)left <= (float)right;

        /// <summary>Compares two values numerically.</summary>
        public static bool operator >=(BFloat16 left, BFloat16 right) => (float)left >= (float)right;
    }
}
