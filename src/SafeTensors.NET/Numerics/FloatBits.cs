using System.Runtime.CompilerServices;

namespace SafeTensors.Internal
{
    /// <summary>
    /// Reinterprets a <see cref="float"/> as its IEEE 754 bit pattern and back.
    /// </summary>
    /// <remarks>
    /// <c>BitConverter.SingleToUInt32Bits</c> only exists from .NET 6, and the
    /// <c>GetBytes</c> route allocates. On netstandard2.0 this falls back to a pointer
    /// reinterpret, which is exactly what the framework method compiles to anyway.
    /// </remarks>
    internal static class FloatBits
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint ToUInt32(float value)
        {
#if NET6_0_OR_GREATER
            return System.BitConverter.SingleToUInt32Bits(value);
#else
            unsafe
            {
                return *(uint*)&value;
            }
#endif
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ToSingle(uint bits)
        {
#if NET6_0_OR_GREATER
            return System.BitConverter.UInt32BitsToSingle(bits);
#else
            unsafe
            {
                return *(float*)&bits;
            }
#endif
        }
    }
}
