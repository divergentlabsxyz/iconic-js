// Unigram tokenizer that reproduces the Hugging Face `tokenizers` output for
// Iconic's tokenizer.json: the precompiled SentencePiece character map, the
// replace and strip normalizers, Metaspace, and Unigram with fused unknowns.

// Rust regex `\s` (Unicode White_Space). JS `\s` differs by two characters.
const WS = "\\t\\n\\v\\f\\r \\u0085\\u00A0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000";
const UNK_PENALTY = 10.0;

const encoder = new TextEncoder();
const decoder = new TextDecoder();

function base64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** SentencePiece's precompiled normalization map (a double-array trie). */
class CharsMap {
  constructor(b64) {
    const bytes = base64ToBytes(b64);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const trieSize = view.getUint32(0, true);
    this.trie = new Uint32Array(trieSize / 4);
    for (let i = 0; i < this.trie.length; i++) this.trie[i] = view.getUint32(4 + i * 4, true);
    this.normalized = bytes.subarray(4 + trieSize);
    this.cache = new Map();
    this.segmenter = new Intl.Segmenter(undefined, { granularity: "grapheme" });
  }

  firstMatch(bytes) {
    const a = this.trie;
    let pos = 0;
    let unit = a[pos];
    pos ^= offset(unit);
    for (const c of bytes) {
      if (c === 0) break;
      pos ^= c;
      if (pos >= a.length) return -1;
      unit = a[pos];
      if (label(unit) !== c) return -1;
      pos ^= offset(unit);
      if ((unit >>> 8) & 1) return a[pos] & 0x7fffffff;
    }
    return -1;
  }

  transform(chunk) {
    let hit = this.cache.get(chunk);
    if (hit !== undefined) return hit;
    const start = this.firstMatch(encoder.encode(chunk));
    if (start < 0) {
      hit = null;
    } else {
      let end = start;
      while (end < this.normalized.length && this.normalized[end] !== 0) end++;
      hit = decoder.decode(this.normalized.subarray(start, end));
    }
    this.cache.set(chunk, hit);
    return hit;
  }

  normalize(text) {
    let out = "";
    for (const { segment } of this.segmenter.segment(text)) {
      if (encoder.encode(segment).length < 6) {
        const norm = this.transform(segment);
        if (norm !== null) {
          out += norm;
          continue;
        }
      }
      for (const ch of segment) {
        const norm = this.transform(ch);
        out += norm !== null ? norm : ch;
      }
    }
    return out;
  }
}

function label(unit) {
  return unit & 0x800000ff;
}

function offset(unit) {
  return (unit >>> 10) << ((unit & (1 << 9)) >>> 6);
}

function buildNormalizer(spec) {
  if (!spec) return (s) => s;
  switch (spec.type) {
    case "Sequence": {
      const steps = spec.normalizers.map(buildNormalizer);
      return (s) => steps.reduce((acc, f) => f(acc), s);
    }
    case "Precompiled": {
      const map = new CharsMap(spec.precompiled_charsmap);
      return (s) => map.normalize(s);
    }
    case "Replace": {
      if (spec.pattern.String !== undefined) {
        const needle = spec.pattern.String;
        return (s) => s.split(needle).join(spec.content);
      }
      const re = new RegExp(spec.pattern.Regex.replaceAll("\\s", `[${WS}]`), "gu");
      return (s) => s.replace(re, spec.content);
    }
    case "Strip": {
      const left = new RegExp(`^[${WS}]+`, "u");
      const right = new RegExp(`[${WS}]+$`, "u");
      return (s) => {
        if (spec.strip_left) s = s.replace(left, "");
        if (spec.strip_right) s = s.replace(right, "");
        return s;
      };
    }
    case "NFC":
    case "NFD":
    case "NFKC":
    case "NFKD":
      return (s) => s.normalize(spec.type);
    default:
      throw new Error(`Unsupported normalizer: ${spec.type}`);
  }
}

export class Tokenizer {
  constructor(json) {
    const model = json.model;
    if (model.type !== "Unigram") throw new Error(`Unsupported model: ${model.type}`);
    const pre = json.pre_tokenizer;
    if (!pre || pre.type !== "Metaspace" || pre.split) {
      throw new Error("Expected a non-splitting Metaspace pre-tokenizer");
    }

    this.normalize = buildNormalizer(json.normalizer);
    this.replacement = pre.replacement;
    this.prepend = pre.prepend_scheme !== "never";
    this.unkId = model.unk_id;

    this.ids = new Map();
    this.scores = new Float64Array(model.vocab.length);
    this.minScore = Infinity;
    this.maxLen = 1;
    model.vocab.forEach(([piece, score], id) => {
      this.ids.set(piece, id);
      this.scores[id] = score;
      if (score < this.minScore) this.minScore = score;
      const len = [...piece].length;
      if (len > this.maxLen) this.maxLen = len;
    });
  }

  /** Token strings for `text`. */
  tokenize(text) {
    let s = this.normalize(text);
    if (s.length === 0) return [];
    s = s.replaceAll(" ", this.replacement);
    if (this.prepend && !s.startsWith(this.replacement)) s = this.replacement + s;
    return this.unigram(s);
  }

  /** Token ids for `text`. */
  encode(text) {
    return this.tokenize(text).map((t) => this.ids.get(t) ?? this.unkId);
  }

  unigram(s) {
    const chars = Array.from(s);
    const n = chars.length;
    const unkScore = this.minScore - UNK_PENALTY;
    const bestScore = new Float64Array(n + 1);
    const startsAt = new Int32Array(n + 1).fill(-1);
    const nodeId = new Int32Array(n + 1);

    const offer = (from, to, id, score) => {
      const cand = score + bestScore[from];
      if (startsAt[to] === -1 || cand > bestScore[to]) {
        bestScore[to] = cand;
        startsAt[to] = from;
        nodeId[to] = id;
      }
    };

    for (let i = 0; i < n; i++) {
      let hasSingle = false;
      let piece = "";
      for (let len = 1; len <= this.maxLen && i + len <= n; len++) {
        piece += chars[i + len - 1];
        const id = this.ids.get(piece);
        if (id === undefined) continue;
        offer(i, i + len, id, this.scores[id]);
        if (len === 1) hasSingle = true;
      }
      if (!hasSingle) offer(i, i + 1, this.unkId, unkScore);
    }

    const out = [];
    let pendingUnk = [];
    for (let end = n; end > 0; ) {
      const start = startsAt[end];
      const piece = chars.slice(start, end).join("");
      if (nodeId[end] === this.unkId) {
        pendingUnk.push(piece);
      } else {
        if (pendingUnk.length) out.push(pendingUnk.reverse().join(""));
        pendingUnk = [];
        out.push(piece);
      }
      end = start;
    }
    if (pendingUnk.length) out.push(pendingUnk.reverse().join(""));
    return out.reverse();
  }
}
