#!/usr/bin/env python3
"""Spec tests for the aPLib decompression ceiling.

These tests fail while the unpacker will grow its output without a bound.
They pass once both copies of the unpacker (tools/mtlib/aplib.py and
docs/flasher/builder.js) refuse to append a literal or a match that would
make the output longer than MAX_UNPACKED, and once tools/build.py rejects a
SysEx file that is not the stock image before it calls depack.

MAX_UNPACKED is 4 MiB. That is above the stock MAIN OS (device.json
section_len 1,744,192) and above the largest image build.py will emit
(END_LIMIT 0x40200000 minus BASE 0x40000400 = 2,096,128). An output of
exactly 4 MiB must still unpack. One byte more must raise:

    ValueError: decompressed output exceeds 4194304 bytes

The same text is required from docs/flasher/builder.js (Error).

Run:
    python3 tests/test_aplib_unpack_limit.py
    node tests/test_aplib_unpack_limit.js
"""
import contextlib
import importlib.util
import io
import json
import pathlib
import subprocess
import sys
import tracemalloc
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from mtlib import aplib, container          # noqa: E402
from mtlib.syx import BYTES_PER_MSG, unwrap, wrap  # noqa: E402
import build                                # noqa: E402

MAX_UNPACKED = 4 * 1024 * 1024
LIMIT_ERROR = f"decompressed output exceeds {MAX_UNPACKED} bytes"
# A match that is applied up to half the ceiling peaks near 2x that size in
# tracemalloc (bytearray resize keeps the old buffer). A full oversized
# unpack peaks near 2x MAX_UNPACKED. 6 MiB sits between those two.
HALF_APPLIED_PEAK = 6 * 1024 * 1024
# A match rejected before it is copied allocates only the stream.
REJECTED_MATCH_PEAK = 1024 * 1024

DEVICE = ROOT / "tweaks" / "model-cycles_OS1.13" / "device.json"
STOCK_SYX_SHA = json.loads(DEVICE.read_text(encoding="utf-8"))["stock_syx_sha256"]


def overlap_stream(copies, extra_literal=False):
    """One literal 0x41, then an overlapping match that repeats it `copies` times."""
    w = aplib._Writer()
    w.literal(0x41)
    w.match(1, copies, 1)
    if extra_literal:
        w.literal(0x42)
    w.end()
    return bytes(w.o)


def repeat_stream(copies, times):
    w = aplib._Writer()
    w.literal(0x41)
    last = 1
    for _ in range(times):
        w.match(1, copies, last)
        last = 1
    w.end()
    return bytes(w.o)


def nonoverlapping_cross_stream():
    """Prefix of MAX/2+1 bytes, then a non-overlapping copy of that whole prefix.

    The copy takes the `off >= n` branch (a slice assign), and the result
    would be two bytes past the ceiling.
    """
    prefix = MAX_UNPACKED // 2 + 1
    w = aplib._Writer()
    w.literal(0x41)
    w.match(1, prefix - 1, 1)
    w.match(prefix, prefix, 1)
    w.end()
    return bytes(w.o)


def syx_with_section3(payload):
    """Checksum-valid Model:Cycles SysEx whose only section is this aPLib stream."""
    off = 0x30
    blob = bytearray(off + len(payload))
    blob[0:4] = b"ELE3"
    blob[8:16] = b"1.13\x00\x00\x00\x00"
    blob[0x1C:0x20] = (1).to_bytes(4, "big")
    blob[0x20:0x24] = (3).to_bytes(4, "big")
    blob[0x24:0x28] = off.to_bytes(4, "big")
    blob[0x28:0x2C] = len(payload).to_bytes(4, "big")
    blob[off:] = payload
    checksum = container.content_checksum(bytes(blob))
    stream = bytearray(8 + len(blob))
    stream[0:4] = len(blob).to_bytes(4, "big")
    stream[4:8] = checksum.to_bytes(4, "big")
    stream[8:] = blob
    if len(stream) % BYTES_PER_MSG:
        stream += b"\0" * (BYTES_PER_MSG - len(stream) % BYTES_PER_MSG)
    return wrap(bytes(stream), 0x11)


def unpack(stream):
    """Run the real depack. Returns (ValueError or None, output length or None, peak)."""
    tracemalloc.start()
    try:
        try:
            data, _ops = aplib.depack(stream)
        except ValueError as exc:
            return exc, None, tracemalloc.get_traced_memory()[1]
        return None, len(data), tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DepackCeilingTests(unittest.TestCase):
    def test_short_overlapping_and_nonoverlapping_matches_roundtrip(self):
        data, _ = aplib.depack(overlap_stream(10))
        self.assertEqual(data, b"A" * 11)
        prefix = 100
        w = aplib._Writer()
        w.literal(0x41)
        w.match(1, prefix - 1, 1)
        w.match(prefix, prefix, 1)
        w.end()
        data, ops = aplib.depack(bytes(w.o))
        self.assertEqual(data, b"A" * (prefix * 2))
        self.assertEqual(ops[-1], (aplib.MATCH, prefix, prefix, prefix))

    def test_exact_ceiling_still_unpacks(self):
        data, _ = aplib.depack(overlap_stream(MAX_UNPACKED - 1))
        self.assertEqual(data, b"A" * MAX_UNPACKED)

    def _assert_refused(self, stream, peak_limit):
        exc, length, peak = unpack(stream)
        self.assertIsNone(length, f"depack returned {length} bytes from {len(stream)} input bytes")
        self.assertIsInstance(exc, ValueError)
        self.assertEqual(str(exc), LIMIT_ERROR)
        self.assertLess(peak, peak_limit, f"output was copied before the ceiling check (peak {peak})")

    def test_overlapping_match_past_ceiling_is_refused_before_copy(self):
        stream = overlap_stream(MAX_UNPACKED)
        self.assertLess(len(stream), 64)
        self._assert_refused(stream, REJECTED_MATCH_PEAK)

    def test_repeated_matches_cannot_sum_past_the_ceiling(self):
        # Each match is half the ceiling, so a per-match cap at the ceiling
        # still lets the second match carry the output over.
        self._assert_refused(repeat_stream(MAX_UNPACKED // 2, 2), HALF_APPLIED_PEAK)

    def test_literal_that_crosses_the_ceiling_is_refused(self):
        self._assert_refused(overlap_stream(MAX_UNPACKED - 1, extra_literal=True), HALF_APPLIED_PEAK * 2)

    def test_nonoverlapping_copy_that_crosses_the_ceiling_is_refused_before_copy(self):
        self._assert_refused(nonoverlapping_cross_stream(), HALF_APPLIED_PEAK)


class HostileSyxEntryTests(unittest.TestCase):
    def setUp(self):
        self._argv = sys.argv[:]
        self._depack = aplib.depack
        self._read = pathlib.Path.read_bytes
        self.payload = overlap_stream(MAX_UNPACKED)
        self.syx = syx_with_section3(self.payload)
        stream, info = unwrap(self.syx)
        self.assertEqual(info["name"], "Model:Cycles")
        parsed = container.parse(stream)
        section = next(s for s in parsed["sections"] if s["id"] == 3)
        got = parsed["blob"][section["off"]:section["off"] + section["size"]]
        self.assertEqual(got, self.payload)

    def tearDown(self):
        sys.argv = self._argv
        aplib.depack = self._depack
        pathlib.Path.read_bytes = self._read

    def _watch_depack(self):
        seen = []

        def wrapped(stream):
            seen.append(stream)
            return self._depack(stream)

        aplib.depack = wrapped
        return seen

    def test_syx_checksums_and_container_parse_before_section_hash(self):
        # setUp already unwraps and parses. The stock hash is not this file.
        import hashlib
        self.assertNotEqual(hashlib.sha256(self.syx).hexdigest(), STOCK_SYX_SHA)

    def test_build_rejects_non_stock_syx_before_depack(self):
        seen = self._watch_depack()
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            src = pathlib.Path(tmp) / "evil.syx"
            out = pathlib.Path(tmp) / "out.syx"
            src.write_bytes(self.syx)
            sys.argv = ["build.py", "-i", str(src), "--os", "-o", str(out)]
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    build.main()
            self.assertNotEqual(caught.exception.code, 0)
            self.assertEqual(seen, [], "build.py called depack before rejecting a non-stock file")
            self.assertFalse(out.exists(), "build.py wrote an output file for a file it refuses")

    def test_build_does_not_depack_a_second_read_of_the_path(self):
        """The bytes that pass the stock-hash check are the bytes that are unpacked.

        A second read of the path between the check and depack must not be
        what gets decompressed. The first read is not the stock image, so
        depack is not reached at all.
        """
        import tempfile
        bomb = self.syx
        other = syx_with_section3(overlap_stream(MAX_UNPACKED + 1000))
        reads = []
        seen = self._watch_depack()

        def read_bytes(path):
            if path == target:
                reads.append(path)
                return bomb if len(reads) == 1 else other
            return self._read(path)

        with tempfile.TemporaryDirectory() as tmp:
            target = pathlib.Path(tmp) / "evil.syx"
            target.write_bytes(bomb)
            out = pathlib.Path(tmp) / "out.syx"
            pathlib.Path.read_bytes = read_bytes
            sys.argv = ["build.py", "-i", str(target), "--os", "-o", str(out)]
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    build.main()
            self.assertNotEqual(caught.exception.code, 0)
            self.assertEqual(len(reads), 1, "build.py read the .syx path again after the first read")
            self.assertEqual(seen, [])
            self.assertFalse(out.exists())

    def _assert_entry_refuses_without_oversized_output(self, call):
        seen = []

        def wrapped(stream):
            tracemalloc.start()
            try:
                try:
                    data, ops = self._depack(stream)
                except ValueError as exc:
                    peak = tracemalloc.get_traced_memory()[1]
                    self.assertEqual(str(exc), LIMIT_ERROR)
                    self.assertLess(peak, REJECTED_MATCH_PEAK, f"peak {peak} while refusing {len(stream)} input bytes")
                    raise
                else:
                    seen.append(len(data))
                    return data, ops
            finally:
                tracemalloc.stop()

        aplib.depack = wrapped
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                call()
        except ValueError as exc:
            self.assertEqual(str(exc), LIMIT_ERROR)
        except SystemExit as exc:
            self.assertNotEqual(exc.code, 0)
        else:
            self.fail(f"entry point returned after unpacking {seen}")
        self.assertFalse(any(n > MAX_UNPACKED for n in seen), f"unpack returned {seen}")

    def test_mainos_extract_does_not_expand_a_hostile_section(self):
        mainos = load_script(
            "mainos_limit",
            ROOT / ".agents/skills/mainos-reverse-engineering/scripts/mainos.py",
        )
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            src = pathlib.Path(tmp) / "evil.syx"
            out = pathlib.Path(tmp) / "mainos.bin"
            src.write_bytes(self.syx)
            sys.argv = ["mainos.py", "extract", str(src), "-o", str(out)]
            self._assert_entry_refuses_without_oversized_output(lambda: mainos.main())
            self.assertFalse(out.exists(), "mainos extract wrote a file for a file it refuses")

    def test_ref_mainos_does_not_expand_a_hostile_section(self):
        ref = load_script("ref_mainos_limit", ROOT / "tools/ref_mainos.py")
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            src = pathlib.Path(tmp) / "evil.syx"
            src.write_bytes(self.syx)
            self._assert_entry_refuses_without_oversized_output(lambda: ref.main_os(src))

    def test_hw_periph_refs_does_not_expand_a_hostile_section(self):
        hw = load_script("hw_periph_limit", ROOT / "tools/hw_periph_refs.py")
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            src = pathlib.Path(tmp) / "evil.syx"
            src.write_bytes(self.syx)
            self._assert_entry_refuses_without_oversized_output(lambda: list(hw.sections(src)))

    def test_javascript_unpacker(self):
        script = ROOT / "tests" / "test_aplib_unpack_limit.js"
        proc = subprocess.run(
            ["node", str(script)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-40:])
            self.fail(f"node {script.name} exited {proc.returncode}\n{tail}")


if __name__ == "__main__":
    unittest.main()
