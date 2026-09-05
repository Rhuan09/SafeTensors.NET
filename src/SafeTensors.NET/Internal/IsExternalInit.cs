#if NETSTANDARD2_0
using System.ComponentModel;

namespace System.Runtime.CompilerServices
{
    /// <summary>
    /// Compiler-required marker enabling <c>init</c> accessors on frameworks that predate them.
    /// </summary>
    [EditorBrowsable(EditorBrowsableState.Never)]
    internal static class IsExternalInit
    {
    }
}
#endif
