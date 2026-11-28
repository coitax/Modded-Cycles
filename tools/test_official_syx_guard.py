#!/usr/bin/env python3
"""The firmware builders must refuse a SysEx that is not the pinned official file.

tools/build.py treats a file as the official OS when the full-file SHA-256 misses
stock_syx_sha256 but the decompressed main program matches section_sha256. It then
re-signs a container that still holds every other section of that file. The browser
builder (docs/flasher/builder.js) does the same check inside build().

These tests build a stock container and a second container with the same main
program and different bootstrap, updater and meta sections, pin device.json at the
stock file, and run the builders. They fail while a builder accepts the second
file and copies those sections into its output. They pass once that file is
refused and no output file is written.

The official Elektron image is not in this repository, so the pins in the temporary
device.json are the hashes of the stock container this file builds. The comparison,
the fallback, the patch and the copy are the builders' own code. The main program
carries the old bytes of the boot-anim tweak so the write step can finish.

Run:
    python3 tools/test_official_syx_guard.py
"""
import contextlib
import hashlib
import hmac
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import build  # noqa: E402
from mtlib import aplib, container  # noqa: E402
from mtlib.syx import BYTES_PER_MSG, unwrap, wrap  # noqa: E402

TWEAK_ID = "boot-anim"
TWEAK_FILE = "43-boot-anim.json"
# Sections other than the main program. Ids match the OS 1.13 container.
META, BOOT, MAIN, UPDATER = 5, 2, 3, 4


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def ops_for(data):
    """Literal/match ops whose repack round-trips `data`. Long runs of one byte become one match."""
    ops = []
    i, n = 0, len(data)
    while i < n:
        if i > 0 and data[i] == data[i - 1]:
            j = i + 1
            while j < n and data[j] == data[i - 1]:
                j += 1
            if j - i >= 2:
                ops.append((aplib.MATCH, i, 1, j - i))
                i = j
                continue
        ops.append((aplib.LITERAL, i, 0, 1))
        i += 1
    return ops


def compress(data):
    blob = aplib.repack(data, ops_for(data), bytearray(len(data)))
    back, _ops = aplib.depack(blob)
    if back != data:
        raise RuntimeError("synthetic section did not survive aPLib")
    return blob


def derive_key(text, const):
    """Same derivation as mtlib.container.find_key: SHA-256(s) XOR SHA-256(reverse(s)) XOR C."""
    h = hashlib.sha256(text).digest()
    hr = hashlib.sha256(text[::-1]).digest()
    return bytes(a ^ b ^ c for a, b, c in zip(h, hr, const))


def assemble(sections, key):
    """ELE3 container and SysEx, laid out the way container.rebuild writes one."""
    n = len(sections)
    pos = (container.TABLE_OFF + n * container.ENTRY_SZ + container.ALIGN - 1) & ~(container.ALIGN - 1)
    blob = bytearray(pos)
    blob[0:4] = container.MAGIC
    blob[8:16] = b"1.13\0\0\0\0"
    blob[container.COUNT_OFF:container.COUNT_OFF + 4] = n.to_bytes(4, "big")
    for index, (sid, data, attr) in enumerate(sections):
        pos = (pos + container.ALIGN - 1) & ~(container.ALIGN - 1)
        if len(blob) < pos:
            blob.extend(b"\0" * (pos - len(blob)))
        entry = container.TABLE_OFF + index * container.ENTRY_SZ
        blob[entry:entry + 4] = sid.to_bytes(4, "big")
        blob[entry + 4:entry + 8] = pos.to_bytes(4, "big")
        blob[entry + 8:entry + 12] = len(data).to_bytes(4, "big")
        blob[entry + 12:entry + 16] = attr.to_bytes(4, "big")
        blob[pos:pos + len(data)] = data
        pos += len(data)
    hmac_off = (pos + 4 + container.ALIGN - 1) & ~(container.ALIGN - 1)
    if len(blob) < hmac_off:
        blob.extend(b"\0" * (hmac_off - len(blob)))
    digest = hmac.new(key, bytes(blob[:hmac_off]), hashlib.sha256).digest()
    blob[hmac_off:hmac_off + container.DIGEST_LEN] = digest
    blob = bytes(blob[:hmac_off + container.DIGEST_LEN])
    stream = container.build_stream(blob, BYTES_PER_MSG)
    return wrap(stream, 0x11, start_seq=114)


def stored_sections(raw):
    stream, _info = unwrap(raw)
    parsed = container.parse(stream)
    return {s["id"]: parsed["blob"][s["off"]:s["off"] + s["size"]] for s in parsed["sections"]}


def recovered_key(raw):
    """The key build.py recovers: depack every section, then container.find_key."""
    stream, _info = unwrap(raw)
    parsed = container.parse(stream)
    plain = []
    for s in parsed["sections"]:
        try:
            plain.append(aplib.depack(parsed["blob"][s["off"]:s["off"] + s["size"]])[0])
        except Exception:
            pass
    message = parsed["blob"][:len(parsed["blob"]) - container.DIGEST_LEN]
    expect = parsed["blob"][len(parsed["blob"]) - container.DIGEST_LEN:]
    return container.find_key(plain, message, expect)


class OfficialSyxGuard(unittest.TestCase):
    """Refusal of a container whose only official part is the main program."""

    @classmethod
    def setUpClass(cls):
        tweak = json.loads((ROOT / "tweaks" / "model-cycles_OS1.13" / TWEAK_FILE).read_text(encoding="utf-8"))
        if tweak["id"] != TWEAK_ID or tweak.get("append") or tweak.get("conflicts") or tweak.get("requires"):
            raise RuntimeError("boot-anim is no longer a single self-contained write tweak")
        write = tweak["writes"][0]
        old = bytes.fromhex(write["old"])
        end = write["off"] + len(old)
        text = b"REVERB SEND"
        const = bytes((i * 13 + 5) & 0xFF for i in range(32))
        material = container.KEY_ANCHOR + text + b"\0" + const
        main = bytearray(b"\x11" * end)
        main[16:16 + len(material)] = material
        main[write["off"]:end] = old
        main = bytes(main)
        if sha256(main) == json.loads(
                (ROOT / "tweaks" / "model-cycles_OS1.13" / "device.json").read_text(encoding="utf-8"))["section_sha256"]:
            raise RuntimeError("synthetic main collided with the pinned official hash")
        cls.key = derive_key(text, const)
        cls.main_stored = compress(main)
        cls.main_sha = sha256(main)
        cls.stock_parts = {
            BOOT: compress(b"BOOT-STOCK\n" + b"S" * 64),
            UPDATER: compress(b"UPDATER-STOCK\n" + b"U" * 64),
            META: b"META-STOCK!!!!!",
        }
        cls.attack_parts = {
            BOOT: compress(b"BOOT-ATTACK\n" + b"A" * 64),
            UPDATER: compress(b"UPDATER-ATTACK\n" + b"X" * 64),
            META: b"META-ATTACK!!!!",
        }
        for sid in (BOOT, UPDATER, META):
            if cls.stock_parts[sid] == cls.attack_parts[sid]:
                raise RuntimeError(f"section {sid} did not change between the two files")
        cls.stock_raw = cls._syx(cls.stock_parts)
        cls.attack_raw = cls._syx(cls.attack_parts)
        if sha256(cls.stock_raw) == sha256(cls.attack_raw):
            raise RuntimeError("stock and substituted files hash the same")
        if recovered_key(cls.attack_raw) != cls.key or recovered_key(cls.stock_raw) != cls.key:
            raise RuntimeError("find_key did not recover the main-program key")
        cls.stock_stored = stored_sections(cls.stock_raw)
        cls.attack_stored = stored_sections(cls.attack_raw)
        if cls.stock_stored[MAIN] != cls.attack_stored[MAIN]:
            raise RuntimeError("the two files do not share one stored main program")

        cls.tmp = pathlib.Path(tempfile.mkdtemp(prefix="syx-guard-"))
        tw = cls.tmp / "model-cycles_OS1.13"
        tw.mkdir()
        src = ROOT / "tweaks" / "model-cycles_OS1.13"
        for f in src.glob("[0-9]*.json"):
            os.symlink(f, tw / f.name)
        device = json.loads((src / "device.json").read_text(encoding="utf-8"))
        device["stock_syx_sha256"] = sha256(cls.stock_raw)
        device["section_sha256"] = cls.main_sha
        cls.device_path = tw / "device.json"
        cls.device_path.write_text(json.dumps(device), encoding="utf-8")
        cls.tweaks_root = cls.tmp / "tweaks"
        cls.tweaks_root.mkdir()
        os.symlink(tw, cls.tweaks_root / tw.name)
        cls._saved = (build.TWEAKS, build.ROOT)
        build.TWEAKS = cls.tweaks_root

    @classmethod
    def tearDownClass(cls):
        build.TWEAKS, build.ROOT = cls._saved
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def _syx(cls, parts):
        sections = [
            (META, parts[META], 0),
            (BOOT, parts[BOOT], 0x04000000),
            (MAIN, cls.main_stored, 0x40000400),
            (UPDATER, parts[UPDATER], 0x80000400),
        ]
        return assemble(sections, cls.key)

    def _run(self, argv, read_follow=None):
        """Run build.main. read_follow, if set, is written over the input after the first read."""
        written = []
        real_read = pathlib.Path.read_bytes
        real_write = pathlib.Path.write_bytes
        followed = {"done": False}

        def read_bytes(self):
            data = real_read(self)
            if read_follow is not None and not followed["done"] and pathlib.Path(self).resolve() == read_follow[0]:
                followed["done"] = True
                real_write(self, read_follow[1])
            return data

        def write_bytes(self, data):
            written.append((pathlib.Path(self).resolve(), bytes(data)))
            return real_write(self, data)

        buf = io.StringIO()
        code = 0
        with unittest.mock.patch.object(pathlib.Path, "read_bytes", read_bytes), \
                unittest.mock.patch.object(pathlib.Path, "write_bytes", write_bytes), \
                contextlib.redirect_stdout(buf):
            try:
                sys.argv = ["build.py", *argv]
                build.main()
            except SystemExit as exc:
                code = 0 if exc.code in (None, 0) else exc.code
        return code, buf.getvalue(), written

    def _assert_refused(self, code, out, output, before, written, label):
        copied = {}
        if output.exists():
            got = stored_sections(output.read_bytes())
            copied = {sid: got.get(sid) == self.attack_stored[sid] for sid in (BOOT, UPDATER, META)}
        created = sorted(str(p.relative_to(output.parent)) for p in output.parent.rglob("*")
                         if p not in before and p.is_file())
        if code not in (0, None) and code != 0:
            self.assertEqual(written, [], f"{label}: refused, but wrote {[str(p) for p, _ in written]}")
            self.assertFalse(output.exists(), f"{label}: refused, but wrote {output}")
            self.assertEqual(created, [], f"{label}: refused, but created {created}")
            self.assertNotIn("\n  ecrit :", out, f"{label}: refused, but reported an output file")
            return
        self.fail(
            f"{label}: the builder accepted a SysEx whose full-file hash is not the pinned official "
            f"file and whose other sections differ. exit={code!r} copied={copied} created={created} "
            f"writes={[str(p) for p, _ in written]} stdout={out[-800:]!r}"
        )

    def test_dev_build_rejects_substituted_bootstrap_and_updater(self):
        work = self.tmp / "dev"
        work.mkdir()
        src = work / "in.syx"
        src.write_bytes(self.attack_raw)
        out = work / "out.syx"
        before = set(work.rglob("*"))
        code, text, written = self._run(["-i", str(src), "-t", TWEAK_ID, "-o", str(out)])
        self.assertEqual(src.read_bytes(), self.attack_raw)
        self._assert_refused(code, text, out, before, written, "-t boot-anim")

    def test_os_build_rejects_substituted_bootstrap_and_updater(self):
        # --os reads os.json after the hash fallback and copies sections in the same
        # rebuild as -t. The recipe here is only boot-anim: the full os.json list also
        # appends payloads from the official Syntakt file, which this tree does not have,
        # and that step does not look at the bootstrap or the updater.
        work = self.tmp / "os"
        work.mkdir()
        recipe = work / "os.json"
        recipe.write_text(json.dumps({
            "name": "Modded Cycles OS",
            "version": "2.0",
            "tweaks": [TWEAK_ID],
        }), encoding="utf-8")
        src = work / "in.syx"
        src.write_bytes(self.attack_raw)
        out = work / "out.syx"
        saved_root = build.ROOT
        build.ROOT = work
        before = set(work.rglob("*"))
        try:
            code, text, written = self._run(["-i", str(src), "--os", "-o", str(out)])
        finally:
            build.ROOT = saved_root
        self.assertEqual(src.read_bytes(), self.attack_raw)
        self._assert_refused(code, text, out, before, written, "--os")

    def test_second_read_of_the_input_is_not_what_gets_built(self):
        """A check of the first read must not be followed by a fresh read of the path.

        The file on disk is the pinned official container for the first read. As soon as
        that read returns, the path is replaced with the substituted container. The bytes
        that were hashed are the ones that may be packed; a later read of the path is the
        substituted file.
        """
        work = self.tmp / "reread"
        work.mkdir()
        src = work / "in.syx"
        src.write_bytes(self.stock_raw)
        out = work / "out.syx"
        code, text, _written = self._run(
            ["-i", str(src), "-t", TWEAK_ID, "-o", str(out)],
            read_follow=(src.resolve(), self.attack_raw),
        )
        if code not in (0, None) and code != 0:
            self.assertFalse(out.exists(), "a refused re-read still left an output file")
            return
        self.assertEqual(code, 0, text)
        self.assertTrue(out.exists(), "the pinned file was readable and the build produced nothing")
        got = stored_sections(out.read_bytes())
        for sid, name in ((BOOT, "bootstrap"), (UPDATER, "updater"), (META, "meta")):
            self.assertEqual(got.get(sid), self.stock_stored[sid],
                             f"section {sid} ({name}) came from a later read of the path, not from the bytes that were hashed")
            self.assertNotEqual(got.get(sid), self.attack_stored[sid],
                                 f"section {sid} ({name}) is the substituted one")

    def test_pinned_official_file_still_builds(self):
        work = self.tmp / "stock"
        work.mkdir()
        src = work / "in.syx"
        src.write_bytes(self.stock_raw)
        out = work / "out.syx"
        code, text, _written = self._run(["-i", str(src), "-t", TWEAK_ID, "-o", str(out)])
        self.assertEqual(code, 0, text)
        self.assertIn("cible :", text)
        self.assertTrue(out.exists())
        got = stored_sections(out.read_bytes())
        for sid in (BOOT, UPDATER, META):
            self.assertEqual(got[sid], self.stock_stored[sid])
        self.assertNotEqual(got[MAIN], self.stock_stored[MAIN])

    def test_browser_builder_rejects_substituted_sections(self):
        script = self.tmp / "browser_guard.js"
        script.write_text(r"""
const fs = require("fs");
const API = require(process.argv[2]);
const device = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const tweak = JSON.parse(fs.readFileSync(process.argv[4], "utf8"));
const stock = new Uint8Array(fs.readFileSync(process.argv[5]));
const attack = new Uint8Array(fs.readFileSync(process.argv[6]));
function sections(raw) {
  const u = API.unwrap(raw);
  const c = API.parseContainer(u.stream);
  const out = {};
  for (const s of c.sections) out[s.id] = Buffer.from(c.blob.subarray(s.off, s.off + s.size));
  return out;
}
let built;
try {
  built = API.build(stock, device, [tweak], {});
} catch (e) {
  console.error("stock file was refused: " + (e && e.message));
  process.exit(2);
}
const got = sections(built.raw);
const want = sections(stock);
for (const id of [2, 4, 5]) {
  if (!got[id] || !want[id] || !got[id].equals(want[id])) {
    console.error("stock build changed section " + id);
    process.exit(2);
  }
}
try {
  const bad = API.build(attack, device, [tweak], {});
  const copied = sections(bad.raw);
  const attackSections = sections(attack);
  const same = [2, 4, 5].filter((id) => copied[id] && attackSections[id] && copied[id].equals(attackSections[id]));
  console.error("browser builder accepted a non-stock SysEx and copied sections " + same.join(","));
  process.exit(1);
} catch (e) {
  process.exit(0);
}
""", encoding="utf-8")
        (self.tmp / "stock-browser.syx").write_bytes(self.stock_raw)
        (self.tmp / "attack-browser.syx").write_bytes(self.attack_raw)
        proc = subprocess.run(
            ["node", str(script), str(ROOT / "docs" / "flasher" / "builder.js"),
             str(self.device_path), str(ROOT / "tweaks" / "model-cycles_OS1.13" / TWEAK_FILE),
             str(self.tmp / "stock-browser.syx"), str(self.tmp / "attack-browser.syx")],
            check=False, capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, (proc.stdout or "") + (proc.stderr or ""))


if __name__ == "__main__":
    unittest.main()
