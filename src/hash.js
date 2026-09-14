// N-gram features and FNV-1a hashing for the lexical stream.
//
// This must match the Python featurizer the model was trained with exactly.
// The lexical table is indexed by these hashes, so a one-bit difference does
// not throw: it quietly turns the stream into noise. test/parity.test.js pins
// it to vectors produced by the Python code.

// Python's str.isspace() set, used by str.strip().
const PY_SPACE = new Set([
  0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0,
  0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007,
  0x2008, 0x2009, 0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000,
]);

// Python's [^\W_]: letters and numbers. Marks are not word characters.
const WORD_RE = /[\p{L}\p{N}]+/gu;

// 64-bit FNV-1a split into two unsigned 32-bit halves, to avoid BigInt.
const OFFSET_HI = 0xcbf29ce4;
const OFFSET_LO = 0x84222325;
export const SEEDS = [
  [0x9e3779b9, 0x7f4a7c15],
  [0xc2b2ae3d, 0x27d4eb4f],
  [0x165667b1, 0x9e3779f9],
];

const encoder = new TextEncoder();

/** Returns the low 32 bits of FNV-1a over the UTF-8 bytes of `text`. */
export function fnv1aLow(bytes, seed) {
  let hi = (OFFSET_HI ^ seed[0]) >>> 0;
  let lo = (OFFSET_LO ^ seed[1]) >>> 0;
  for (let i = 0; i < bytes.length; i++) {
    lo = (lo ^ bytes[i]) >>> 0;
    // h * 0x100000001b3 = h * 0x1b3 + (h << 40), mod 2^64
    const loMul = lo * 0x1b3;
    const carry = Math.floor(loMul / 0x100000000);
    const newHi = (hi * 0x1b3 + carry + ((lo << 8) >>> 0)) % 0x100000000;
    lo = loMul >>> 0;
    hi = newHi >>> 0;
  }
  return lo;
}

function pyStrip(s) {
  const cps = Array.from(s);
  let a = 0;
  let b = cps.length;
  while (a < b && PY_SPACE.has(cps[a].codePointAt(0))) a++;
  while (b > a && PY_SPACE.has(cps[b - 1].codePointAt(0))) b--;
  return cps.slice(a, b).join("");
}

export function normalize(text) {
  return pyStrip(text.normalize("NFC")).toLowerCase();
}

export function features(text, charNgrams = [3, 4, 5], wordNgrams = [1, 2]) {
  const words = normalize(text).match(WORD_RE) ?? [];
  const out = [];
  for (const w of words) {
    const padded = Array.from(`^${w}$`);
    for (const n of charNgrams) {
      for (let i = 0; i + n <= padded.length; i++) {
        out.push(`c${n}:${padded.slice(i, i + n).join("")}`);
      }
    }
  }
  for (const n of wordNgrams) {
    for (let i = 0; i + n <= words.length; i++) {
      out.push(`w${n}:${words.slice(i, i + n).join("_")}`);
    }
  }
  return out;
}

/**
 * Hashes up to `maxFeatures` features into signed bucket indices, two per
 * feature. Signs are scaled by 1/sqrt(feature count).
 */
export function hashFeatures(text, { buckets, nHashes = 2, maxFeatures = 512 }) {
  const mask = buckets - 1;
  const feats = features(text).slice(0, maxFeatures);
  const idx = new Int32Array(feats.length * nHashes);
  const sign = new Float32Array(feats.length * nHashes);
  const norm = 1 / Math.sqrt(Math.max(feats.length, 1));

  let p = 0;
  for (const f of feats) {
    const bytes = encoder.encode(f);
    const s = (fnv1aLow(bytes, SEEDS[SEEDS.length - 1]) >>> 7) & 1 ? norm : -norm;
    for (let k = 0; k < nHashes; k++) {
      idx[p] = fnv1aLow(bytes, SEEDS[k]) & mask;
      sign[p] = s;
      p++;
    }
  }
  return { idx, sign, count: feats.length };
}
