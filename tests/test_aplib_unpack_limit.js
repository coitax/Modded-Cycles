#!/usr/bin/env node
"use strict";
/**
 * Spec tests for the browser copy of the aPLib unpacker (docs/flasher/builder.js).
 *
 * Fails while aplibDepack will grow an output without a bound. Passes once a
 * literal or a match that would make the output longer than MAX throws
 *   Error: decompressed output exceeds 4194304 bytes
 * before those bytes are appended. An output of exactly 4 MiB must still unpack.
 *
 * The child process under a 48 MB heap runs an 8 MB match. That child must
 * throw the error above and exit 0. Dying with an out-of-memory abort means
 * the match was copied before the check.
 *
 *   node tests/test_aplib_unpack_limit.js
 */
const path = require("path");
const { spawnSync } = require("child_process");
const B = require(path.join(__dirname, "..", "docs", "flasher", "builder.js"));

const MAX = 4 * 1024 * 1024;
const LIMIT_ERROR = `decompressed output exceeds ${MAX} bytes`;
const FAR = 3328;

class Writer {
  constructor() {
    this.o = [0, 0, 0, 0, 0, 0, 0, 0];
    this.tagpos = -1;
    this.tagbits = 0;
  }
  bit(v) {
    if (this.tagbits === 0) {
      this.tagpos = this.o.length;
      this.o.push(0);
      this.tagbits = 8;
    }
    if (v) this.o[this.tagpos] |= 1 << (this.tagbits - 1);
    this.tagbits -= 1;
  }
  gamma(v) {
    for (let i = (31 - Math.clz32(v >>> 0)) - 1; i >= 0; i--) {
      this.bit((v >>> i) & 1);
      this.bit(i === 0 ? 1 : 0);
    }
  }
  literal(b) {
    this.bit(1);
    this.o.push(b & 0xff);
  }
  match(off, n, lastOff) {
    this.bit(0);
    if (off === lastOff) this.gamma(2);
    else {
      const raw = off + 767;
      this.gamma(raw >>> 8);
      this.o.push(raw & 0xff);
    }
    const base = n - 1 - (off > FAR ? 1 : 0);
    if (base <= 3) {
      this.bit(base >>> 1);
      this.bit(base & 1);
    } else {
      this.bit(0);
      this.bit(0);
      this.gamma(base - 2);
    }
  }
  end() {
    this.bit(0);
    this.gamma(0x1000002);
    this.o.push(0xff);
  }
  bytes() {
    return Uint8Array.from(this.o);
  }
}

function overlap(copies, extraLiteral) {
  const w = new Writer();
  w.literal(0x41);
  w.match(1, copies, 1);
  if (extraLiteral) w.literal(0x42);
  w.end();
  return w.bytes();
}

function repeats(copies, times) {
  const w = new Writer();
  w.literal(0x41);
  let last = 1;
  for (let i = 0; i < times; i++) {
    w.match(1, copies, last);
    last = 1;
  }
  w.end();
  return w.bytes();
}

function nonoverlappingCross() {
  const prefix = (MAX >>> 1) + 1;
  const w = new Writer();
  w.literal(0x41);
  w.match(1, prefix - 1, 1);
  w.match(prefix, prefix, 1);
  w.end();
  return w.bytes();
}

function be32(n) {
  return Uint8Array.of((n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255);
}

function syxWithSection3(payload) {
  const off = 0x30;
  const blob = new Uint8Array(off + payload.length);
  blob.set([0x45, 0x4c, 0x45, 0x33], 0);
  blob.set(be32(1), 0x1c);
  blob.set(be32(3), 0x20);
  blob.set(be32(off), 0x24);
  blob.set(be32(payload.length), 0x28);
  blob.set(payload, off);
  const stream = new Uint8Array(8 + blob.length);
  stream.set(be32(blob.length), 0);
  stream.set(blob, 8);
  const pad = (101 - (stream.length % 101)) % 101;
  const body = new Uint8Array(stream.length + pad);
  body.set(stream, 0);
  return B.wrap(body, 0x11);
}

function assert(cond, msg) {
  if (!cond) {
    console.error("FAIL", msg);
    process.exit(1);
  }
}

function expectLimit(stream, label) {
  let threw = null;
  let outLen = null;
  try {
    outLen = B.aplibDepack(stream).data.length;
  } catch (e) {
    threw = e;
  }
  assert(threw instanceof Error, `${label}: aplibDepack returned ${outLen} bytes`);
  assert(threw.message === LIMIT_ERROR, `${label}: ${threw.message}`);
}

function oomChild() {
  const stream = overlap(8 * 1024 * 1024);
  expectLimit(stream, "8MB match under a tight heap");
  const raw = syxWithSection3(stream);
  let threw = null;
  try {
    B.build(raw, { section_sha256: "00".repeat(32) }, [], {});
  } catch (e) {
    threw = e;
  }
  assert(threw instanceof Error, "build() returned a firmware from a hostile section");
  assert(
    threw.message === LIMIT_ERROR,
    `build() should refuse while unpacking, got: ${threw.message}`
  );
  console.log("oom-guard threw", LIMIT_ERROR);
}

if (process.argv[2] === "oom-guard") {
  oomChild();
  process.exit(0);
}

const small = overlap(10);
const smallOut = B.aplibDepack(small).data;
assert(smallOut.length === 11 && smallOut.every((b) => b === 0x41), "short match did not round-trip");

const exact = B.aplibDepack(overlap(MAX - 1)).data;
assert(exact.length === MAX && exact[0] === 0x41 && exact[MAX - 1] === 0x41, "output of exactly 4 MiB must unpack");

assert(overlap(MAX).length < 64, "oversized match should be a few dozen compressed bytes");
expectLimit(overlap(MAX), "one overlapping match past the ceiling");
expectLimit(repeats(MAX >>> 1, 2), "two matches that sum past the ceiling");
expectLimit(overlap(MAX - 1, true), "literal past the ceiling");
expectLimit(nonoverlappingCross(), "non-overlapping copy past the ceiling");

const child = spawnSync(
  process.execPath,
  ["--max-old-space-size=48", __filename, "oom-guard"],
  { encoding: "utf8" }
);
if (child.status !== 0) {
  const tail = `${child.stdout || ""}\n${child.stderr || ""}`.trim().split("\n").slice(-15).join("\n");
  console.error(`oom-guard exited ${child.status} (signal ${child.signal})\n${tail}`);
  process.exit(1);
}
assert(
  (child.stdout || "").includes("oom-guard threw"),
  "oom-guard did not report the ceiling error"
);
console.log("ok");
