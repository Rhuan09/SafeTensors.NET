using System.Text;

namespace SafeTensors.Tests.Infrastructure;

/// <summary>
/// Builds SafeTensors byte streams by hand, including ones no honest producer would write.
/// </summary>
/// <remarks>
/// The reader's job is to reject bad files, so most of the interesting tests need a file
/// the writer would refuse to produce. These helpers go straight to bytes.
/// </remarks>
internal static class Fixture
{
    /// <summary>
    /// Assembles a file from a literal JSON header and a data section of
    /// <paramref name="dataBytes"/> zero bytes.
    /// </summary>
    public static byte[] FromJson(string headerJson, int dataBytes, bool pad = false)
    {
        byte[] json = Encoding.UTF8.GetBytes(headerJson);
        int padding = pad ? (8 - (json.Length % 8)) % 8 : 0;

        var buffer = new MemoryStream();
        buffer.Write(BitConverter.GetBytes((ulong)(json.Length + padding)), 0, 8);
        buffer.Write(json, 0, json.Length);

        for (int i = 0; i < padding; i++)
        {
            buffer.WriteByte(0x20);
        }

        for (int i = 0; i < dataBytes; i++)
        {
            buffer.WriteByte(0);
        }

        return buffer.ToArray();
    }

    /// <summary>Declares a header length that does not match the JSON that follows.</summary>
    public static byte[] WithDeclaredHeaderLength(string headerJson, ulong declaredLength, int dataBytes)
    {
        byte[] json = Encoding.UTF8.GetBytes(headerJson);

        var buffer = new MemoryStream();
        buffer.Write(BitConverter.GetBytes(declaredLength), 0, 8);
        buffer.Write(json, 0, json.Length);

        for (int i = 0; i < dataBytes; i++)
        {
            buffer.WriteByte(0);
        }

        return buffer.ToArray();
    }

    /// <summary>Creates a temporary directory that deletes itself.</summary>
    public static TempDirectory NewDirectory() => new();
}

internal sealed class TempDirectory : IDisposable
{
    public TempDirectory()
    {
        Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "safetensors-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path);
    }

    public string Path { get; }

    public string File(string name) => System.IO.Path.Combine(Path, name);

    public void Dispose()
    {
        try
        {
            Directory.Delete(Path, recursive: true);
        }
        catch (IOException)
        {
            // A mapping the test left open keeps the file locked on Windows; the temp
            // directory is not worth failing a green test over.
        }
        catch (UnauthorizedAccessException)
        {
        }
    }
}
