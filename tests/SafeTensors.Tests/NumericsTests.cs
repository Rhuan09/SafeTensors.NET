namespace SafeTensors.Tests;

/// <summary>
/// The 16-bit float types, checked against the framework where the framework has an answer.
/// </summary>
/// <remarks>
/// <see cref="Float16"/> carries its own software conversion so that netstandard2.0 targets
/// keep F16 support. That code is compiled into every build, so running it against
/// <see cref="Half"/> here proves the .NET Framework and Unity path as well — otherwise it
/// would be the one implementation nothing ever exercises.
/// </remarks>
public class NumericsTests
{
    [Fact]
    public void Float16_to_single_matches_Half_for_every_bit_pattern()
    {
        for (int bits = 0; bits <= ushort.MaxValue; bits++)
        {
            float expected = (float)BitConverter.UInt16BitsToHalf((ushort)bits);
            float actual = Float16.BitsToSingle((ushort)bits);

            if (float.IsNaN(expected))
            {
                Assert.True(float.IsNaN(actual), $"0x{bits:X4} should be NaN.");
                continue;
            }

            Assert.True(
                expected.Equals(actual),
                $"0x{bits:X4}: expected {expected}, got {actual}.");
        }
    }

    [Fact]
    public void Single_to_Float16_matches_Half_across_the_whole_range()
    {
        // Every representable half, plus each one nudged toward its neighbours, which is
        // where round-to-nearest-even and the subnormal boundary actually get decided.
        for (int bits = 0; bits <= ushort.MaxValue; bits++)
        {
            float value = (float)BitConverter.UInt16BitsToHalf((ushort)bits);
            if (float.IsNaN(value))
            {
                continue;
            }

            foreach (float candidate in Neighbours(value))
            {
                // NaN payloads are not specified by IEEE 754 and the two implementations
                // pick different ones on purpose: Half truncates the significand, Float16
                // collapses to a canonical quiet NaN so a truncation can never land on
                // infinity. NaN-ness is asserted separately.
                if (float.IsNaN(candidate))
                {
                    Assert.True(new Float16(Float16.SingleToBits(candidate)).IsNaN);
                    continue;
                }

                ushort expected = BitConverter.HalfToUInt16Bits((Half)candidate);
                ushort actual = Float16.SingleToBits(candidate);

                Assert.True(
                    expected == actual,
                    $"{candidate:R}: expected 0x{expected:X4}, got 0x{actual:X4}.");
            }
        }
    }

    [Fact]
    public void Single_to_Float16_matches_Half_on_a_pseudo_random_sample()
    {
        var random = new Random(20260905);

        for (int i = 0; i < 200_000; i++)
        {
            float value = BitConverter.UInt32BitsToSingle((uint)random.NextInt64(uint.MinValue, uint.MaxValue));
            if (float.IsNaN(value))
            {
                continue;
            }


            Assert.Equal(BitConverter.HalfToUInt16Bits((Half)value), Float16.SingleToBits(value));
        }
    }

    [Theory]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    [InlineData(0f)]
    [InlineData(65504f)]      // largest finite half
    [InlineData(65520f)]      // midpoint above it: rounds to infinity
    [InlineData(5.9604645e-8f)] // smallest subnormal half
    [InlineData(2.9802322e-8f)] // exactly half of it: rounds to even, which is zero
    public void Float16_handles_the_boundaries_the_way_Half_does(float value)
    {
        Assert.Equal(BitConverter.HalfToUInt16Bits((Half)value), Float16.SingleToBits(value));
    }

    [Fact]
    public void Float16_keeps_NaN_as_NaN()
    {
        Assert.True(((Float16)float.NaN).IsNaN);
        Assert.True(float.IsNaN((Float16)float.NaN));
    }

    [Fact]
    public void Float16_keeps_the_sign_of_negative_zero()
    {
        Assert.Equal(BitConverter.HalfToUInt16Bits((Half)(-0.0f)), Float16.SingleToBits(-0.0f));
        Assert.Equal(0x8000, Float16.SingleToBits(-0.0f));
    }

    [Fact]
    public void Float16_reinterprets_to_and_from_Half_without_converting()
    {
        for (int bits = 0; bits <= ushort.MaxValue; bits++)
        {
            var value = new Float16((ushort)bits);
            Half half = value;

            Assert.Equal((ushort)bits, BitConverter.HalfToUInt16Bits(half));
            Assert.Equal((ushort)bits, ((Float16)half).RawBits);
        }
    }

    [Fact]
    public void BFloat16_to_single_is_the_top_half_of_the_float()
    {
        for (int bits = 0; bits <= ushort.MaxValue; bits++)
        {
            float actual = new BFloat16((ushort)bits);
            uint expected = (uint)bits << 16;

            if (float.IsNaN(actual))
            {
                continue;
            }

            Assert.Equal(expected, BitConverter.SingleToUInt32Bits(actual));
        }
    }

    [Fact]
    public void BFloat16_round_trips_every_value_it_can_represent()
    {
        for (int bits = 0; bits <= ushort.MaxValue; bits++)
        {
            var value = new BFloat16((ushort)bits);
            if (value.IsNaN)
            {
                continue;
            }

            Assert.Equal((ushort)bits, ((BFloat16)(float)value).RawBits);
        }
    }

    [Fact]
    public void BFloat16_rounds_to_nearest_even()
    {
        // 1.0f has a zero significand, so the next float up is one ulp of binary32. Halfway
        // between two bfloat16 values must land on the even one, not simply truncate.
        float justAbove = BitConverter.UInt32BitsToSingle(0x3F800000u | 0x8000u);

        Assert.Equal(0x3F80, ((BFloat16)justAbove).RawBits);

        float justBelowNext = BitConverter.UInt32BitsToSingle(0x3F810000u | 0x8000u);
        Assert.Equal(0x3F82, ((BFloat16)justBelowNext).RawBits);
    }

    [Fact]
    public void BFloat16_keeps_NaN_as_NaN_rather_than_rounding_it_to_infinity()
    {
        // The rounding bias added to a significand of all ones carries into the exponent,
        // which would silently turn a NaN into an infinity.
        float nan = BitConverter.UInt32BitsToSingle(0x7FFFFFFFu);

        Assert.True(((BFloat16)nan).IsNaN);
        Assert.False(((BFloat16)nan).IsInfinity);
    }

    [Fact]
    public void Both_types_are_exactly_two_bytes()
    {
        Assert.Equal(2, System.Runtime.InteropServices.Marshal.SizeOf<Float16>());
        Assert.Equal(2, System.Runtime.InteropServices.Marshal.SizeOf<BFloat16>());
    }

    private static IEnumerable<float> Neighbours(float value)
    {
        yield return value;

        uint bits = BitConverter.SingleToUInt32Bits(value);
        if (bits != 0)
        {
            yield return BitConverter.UInt32BitsToSingle(bits - 1);
        }

        yield return BitConverter.UInt32BitsToSingle(bits + 1);
    }
}
