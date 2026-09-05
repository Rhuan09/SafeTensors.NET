using System;
using System.Globalization;
using System.Runtime.InteropServices;
using SafeTensors.Internal;

namespace SafeTensors
{
    /// <summary>
    /// IEEE 754 binary16, laid out so that a <c>ReadOnlySpan&lt;Float16&gt;</c> can be cast
    /// directly over F16 tensor bytes on every supported framework.
    /// </summary>
    /// <remarks>
    /// <para>
    /// On .NET 5 and later this is bit-identical to <c>Half</c> and converts to and
    /// from it for free; the implicit operators exist so you can pass either type around.
    /// It is a distinct type only because netstandard2.0 has no <c>Half</c>, and a
    /// tensor library that drops F16 on .NET Framework and Unity is not much of a tensor
    /// library.
    /// </para>
    /// <para>
    /// The bit-level conversions are the same code on every framework and are verified in
    /// the test suite against <c>Half</c> across all 65 536 values, so the
    /// netstandard2.0 build is not a second, less-tested implementation.
    /// </para>
    /// </remarks>
    [StructLayout(LayoutKind.Sequential, Size = 2)]
    public readonly struct Float16 : IEquatable<Float16>, IComparable<Float16>, IFormattable
    {
        private readonly ushort _bits;

        /// <summary>Creates a value from its raw 16-bit encoding.</summary>
        public Float16(ushort rawBits) => _bits = rawBits;

        /// <summary>Gets the raw 16-bit encoding.</summary>
        public ushort RawBits => _bits;

        /// <summary>Gets a value indicating whether this is NaN.</summary>
        public bool IsNaN => (_bits & 0x7C00) == 0x7C00 && (_bits & 0x03FF) != 0;

        /// <summary>Gets a value indicating whether this is positive or negative infinity.</summary>
        public bool IsInfinity => (_bits & 0x7FFF) == 0x7C00;

        /// <summary>Converts to <see cref="float"/>. Always exact.</summary>
        public static implicit operator float(Float16 value) => BitsToSingle(value._bits);

        /// <summary>Converts from <see cref="float"/>, rounding to nearest with ties to even.</summary>
        public static explicit operator Float16(float value) => new Float16(SingleToBits(value));

        /// <summary>Converts to <see cref="double"/>. Always exact.</summary>
        public static implicit operator double(Float16 value) => BitsToSingle(value._bits);

#if NET5_0_OR_GREATER
        /// <summary>Reinterprets a <see cref="Half"/>. Same encoding, no conversion.</summary>
        public static implicit operator Float16(Half value)
            => new Float16(BitConverter.HalfToUInt16Bits(value));

        /// <summary>Reinterprets as a <see cref="Half"/>. Same encoding, no conversion.</summary>
        public static implicit operator Half(Float16 value)
            => BitConverter.UInt16BitsToHalf(value._bits);
#endif

        /// <summary>Converts a raw binary16 encoding to <see cref="float"/>.</summary>
        /// <remarks>Exact for every input: binary32 can represent every binary16 value.</remarks>
        internal static float BitsToSingle(ushort bits)
        {
            uint sign = (uint)(bits & 0x8000) << 16;
            uint exponent = (uint)(bits >> 10) & 0x1Fu;
            uint mantissa = (uint)(bits & 0x03FF);

            if (exponent == 0)
            {
                if (mantissa == 0)
                {
                    return FloatBits.ToSingle(sign);
                }

                // Subnormal. Normalise until the implicit bit reaches position 10, then
                // rebuild as a normal binary32: value = 1.f x 2^(-14 - shift).
                int shift = 0;
                while ((mantissa & 0x0400u) == 0)
                {
                    mantissa <<= 1;
                    shift++;
                }

                mantissa &= 0x03FFu;
                uint subnormalExponent = (uint)(113 - shift);
                return FloatBits.ToSingle(sign | (subnormalExponent << 23) | (mantissa << 13));
            }

            if (exponent == 0x1Fu)
            {
                return FloatBits.ToSingle(sign | 0x7F800000u | (mantissa << 13));
            }

            return FloatBits.ToSingle(sign | ((exponent - 15 + 127) << 23) | (mantissa << 13));
        }

        /// <summary>Converts a <see cref="float"/> to raw binary16, round to nearest even.</summary>
        internal static ushort SingleToBits(float value)
        {
            uint f = FloatBits.ToUInt32(value);
            uint sign = (f >> 16) & 0x8000u;
            uint abs = f & 0x7FFFFFFFu;

            if (abs >= 0x7F800000u)
            {
                // Infinity keeps its encoding; NaN collapses to a canonical quiet NaN
                // rather than a truncated payload that could land on infinity.
                return (ushort)(sign | (abs > 0x7F800000u ? 0x7E00u : 0x7C00u));
            }

            int exponent = (int)(abs >> 23) - 127;

            if (exponent >= 16)
            {
                return (ushort)(sign | 0x7C00u);
            }

            if (exponent >= -14)
            {
                uint mantissa = abs & 0x7FFFFFu;
                uint bits = (uint)((exponent + 15) << 10) | (mantissa >> 13);
                uint remainder = mantissa & 0x1FFFu;

                // A carry out of the mantissa lands in the exponent, and a carry out of
                // the exponent lands on infinity. Both are the correct results.
                if (remainder > 0x1000u || (remainder == 0x1000u && (bits & 1) != 0))
                {
                    bits++;
                }

                return (ushort)(sign | bits);
            }

            if (exponent < -25)
            {
                return (ushort)sign;
            }

            // Subnormal binary16. Restore the implicit bit and round the shifted-out tail.
            uint significand = (abs & 0x7FFFFFu) | 0x800000u;
            int shift = -exponent - 1;
            uint result = significand >> shift;
            uint tail = significand & ((1u << shift) - 1u);
            uint halfway = 1u << (shift - 1);

            if (tail > halfway || (tail == halfway && (result & 1) != 0))
            {
                result++;
            }

            return (ushort)(sign | result);
        }

        /// <inheritdoc />
        public bool Equals(Float16 other) => (float)this == (float)other;

        /// <inheritdoc />
        public override bool Equals(object? obj) => obj is Float16 other && Equals(other);

        /// <inheritdoc />
        public override int GetHashCode() => ((float)this).GetHashCode();

        /// <inheritdoc />
        public int CompareTo(Float16 other) => ((float)this).CompareTo((float)other);

        /// <inheritdoc />
        public override string ToString() => ((float)this).ToString(CultureInfo.CurrentCulture);

        /// <inheritdoc />
        public string ToString(string? format, IFormatProvider? formatProvider)
            => ((float)this).ToString(format, formatProvider);

        /// <summary>Compares two values numerically. NaN is never equal to anything.</summary>
        public static bool operator ==(Float16 left, Float16 right) => (float)left == (float)right;

        /// <summary>Compares two values numerically.</summary>
        public static bool operator !=(Float16 left, Float16 right) => !(left == right);

        /// <summary>Compares two values numerically.</summary>
        public static bool operator <(Float16 left, Float16 right) => (float)left < (float)right;

        /// <summary>Compares two values numerically.</summary>
        public static bool operator >(Float16 left, Float16 right) => (float)left > (float)right;

        /// <summary>Compares two values numerically.</summary>
        public static bool operator <=(Float16 left, Float16 right) => (float)left <= (float)right;

        /// <summary>Compares two values numerically.</summary>
        public static bool operator >=(Float16 left, Float16 right) => (float)left >= (float)right;
    }
}
