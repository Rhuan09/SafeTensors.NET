"""Regenerates the fixtures in this directory.

Two kinds of file live here, produced two different ways on purpose.

The `reference_*` files come from the Hugging Face `safetensors` package, so the
.NET reader is checked against bytes a real producer emitted rather than against
bytes this library itself wrote. A round-trip test only proves the writer and the
reader agree with each other; these prove they agree with everyone else.

The `malformed_*` files are assembled byte by byte, because no honest producer
will emit them and the reference implementation refuses to. They exist so the
reader's rejection paths are exercised against real files on disk, not only
against buffers built in a test.

Usage:

    python -m pip install numpy safetensors ml_dtypes
    python genTests.py
"""

import json
import struct
from pathlib import Path

import ml_dtypes
import numpy as np
from safetensors.numpy import save_file

HERE = Path(__file__).parent

# Chosen so every value is exact in float16, bfloat16, float32 and float64 alike.
# A fixture that needs an epsilon comparison is a fixture that hides rounding bugs.
FLOATS = [-2.0, -0.5, 0.5, 2.0]
SIGNED = [-2, -1, 1, 2]
UNSIGNED = [1, 2, 3, 4]
BOOLS = [True, False, True, False]


def write_reference_all_dtypes() -> None:
    """One tensor per dtype the format defines and numpy can express.

    The 8-bit float types are absent because numpy has no counterpart for them;
    they are covered by the round-trip tests instead.
    """
    tensors = {
        "bool": np.array(BOOLS, dtype=np.bool_),
        "u8": np.array(UNSIGNED, dtype=np.uint8),
        "i8": np.array(SIGNED, dtype=np.int8),
        "i16": np.array(SIGNED, dtype=np.int16),
        "u16": np.array(UNSIGNED, dtype=np.uint16),
        "f16": np.array(FLOATS, dtype=np.float16),
        "bf16": np.array(FLOATS, dtype=ml_dtypes.bfloat16),
        "i32": np.array(SIGNED, dtype=np.int32),
        "u32": np.array(UNSIGNED, dtype=np.uint32),
        "f32": np.array(FLOATS, dtype=np.float32),
        "f64": np.array(FLOATS, dtype=np.float64),
        "i64": np.array(SIGNED, dtype=np.int64),
        "u64": np.array(UNSIGNED, dtype=np.uint64),
    }
    save_file(tensors, HERE / "reference_all_dtypes.safetensors")


def write_reference_metadata() -> None:
    """A `__metadata__` block alongside a tensor."""
    save_file(
        {"weight": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)},
        HERE / "reference_metadata.safetensors",
        metadata={"format": "np", "framework": "reference", "step": "1200"},
    )


def write_reference_edge_shapes() -> None:
    """Shapes that implementations disagree about.

    A rank-0 scalar holds one element, not zero. A tensor with a zero dimension
    holds no elements and occupies no bytes, which makes its start offset equal
    its end offset. Both are legal and both are easy to get wrong.
    """
    save_file(
        {
            "scalar": np.array(42.0, dtype=np.float32),
            "empty": np.zeros((0, 4), dtype=np.float32),
            "rank4": np.arange(2 * 3 * 4 * 5, dtype=np.int32).reshape(2, 3, 4, 5),
        },
        HERE / "reference_edge_shapes.safetensors",
    )


def build(header: dict, data: bytes, *, declared_header_length: int | None = None) -> bytes:
    """Assembles a file from a header and a data section, telling no lies unless asked.

    `declared_header_length` overrides the length prefix so a file can claim a
    header it does not have.
    """
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    length = declared_header_length if declared_header_length is not None else len(encoded)
    return struct.pack("<Q", length) + encoded + data


def write_malformed_duplicate_keys() -> None:
    """The same tensor name defined twice, at different offsets.

    JSON parsers do not agree on which duplicate wins, so a reader that accepts
    this returns different bytes than the next reader does for the same file.
    Built as text because `json.dumps` cannot express a duplicate key.
    """
    header = (
        b'{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},'
        b'"weight":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}'
    )
    payload = struct.pack("<Q", len(header)) + header + bytes(16)
    (HERE / "malformed_duplicate_keys.safetensors").write_bytes(payload)


def write_malformed_overlapping_tensors() -> None:
    """Two tensors whose byte ranges intersect.

    One file handing the same memory to two names, which no producer emits and
    a reader must not accept.
    """
    header = {
        "a": {"dtype": "U8", "shape": [8], "data_offsets": [0, 8]},
        "b": {"dtype": "U8", "shape": [8], "data_offsets": [4, 12]},
    }
    (HERE / "malformed_overlapping_tensors.safetensors").write_bytes(build(header, bytes(16)))


def write_malformed_size_mismatch() -> None:
    """A shape and dtype that do not agree with the byte range they claim.

    [4] of F32 is 16 bytes; this one says 8.
    """
    header = {"weight": {"dtype": "F32", "shape": [4], "data_offsets": [0, 8]}}
    (HERE / "malformed_size_mismatch.safetensors").write_bytes(build(header, bytes(8)))


def write_malformed_header_too_large() -> None:
    """A length prefix asking for far more header than the file contains.

    The prefix is read before anything about the file has been validated, so an
    unbounded value here is an allocation request from an untrusted source.
    """
    header = {"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    (HERE / "malformed_header_too_large.safetensors").write_bytes(
        build(header, bytes(8), declared_header_length=1 << 40)
    )


def write_malformed_empty() -> None:
    """Zero bytes: not even the 8-byte length prefix."""
    (HERE / "malformed_empty.safetensors").write_bytes(b"")


def main() -> None:
    write_reference_all_dtypes()
    write_reference_metadata()
    write_reference_edge_shapes()
    write_malformed_duplicate_keys()
    write_malformed_overlapping_tensors()
    write_malformed_size_mismatch()
    write_malformed_header_too_large()
    write_malformed_empty()

    for path in sorted(HERE.glob("*.safetensors")):
        print(f"{path.stat().st_size:>8} bytes  {path.name}")


if __name__ == "__main__":
    main()
