// Reads the quantised weight bundle and runs the network.
//
// The two large tables are 4-bit palettized. They are decoded one row at a
// time, because an inference touches a few hundred rows out of ~100,000.

export class Weights {
  constructor(manifest, buffer) {
    this.manifest = manifest;
    this.buffer = buffer;
    this.view = new DataView(buffer);
    this.tensors = manifest.tensors;
  }

  shape(name) {
    return this.tensors[name].shape;
  }

  float32(name) {
    const t = this.tensors[name];
    const n = t.shape.reduce((a, b) => a * b, 1);
    return new Float32Array(this.buffer.slice(t.offset, t.offset + n * 4));
  }

  /** Adds `weight * row` of a palettized table into `out`. */
  addRow(name, row, weight, out) {
    const t = this.tensors[name];
    const cols = t.shape[1];
    const nbits = Number(t.dtype.slice(1));
    const levels = 1 << nbits;
    const codeMask = levels - 1;
    const paletteBase = t.offset + t.palette_offset;
    for (let c = 0; c < cols; c++) {
      const bitPos = (row * cols + c) * nbits;
      const byte = this.view.getUint8(t.offset + Math.floor(bitPos / 8));
      const code = (byte >> (8 - nbits - (bitPos % 8))) & codeMask;
      const group = Math.min(Math.floor(c / t.group_size), t.groups - 1);
      out[c] += weight * this.view.getFloat32(paletteBase + (group * levels + code) * 4, true);
    }
  }
}

export class Network {
  constructor(weights) {
    this.w = weights;
    this.lexDim = weights.shape("lex_weight")[1];
    this.semDim = weights.shape("sem_table")[1];
    this.lexNormW = weights.float32("lex_norm.weight");
    this.lexNormB = weights.float32("lex_norm.bias");
    this.semNormW = weights.float32("sem_norm.weight");
    this.semNormB = weights.float32("sem_norm.bias");
    this.fc1 = { w: weights.float32("fuse.0.weight"), b: weights.float32("fuse.0.bias"), shape: weights.shape("fuse.0.weight") };
    this.fc2 = { w: weights.float32("fuse.3.weight"), b: weights.float32("fuse.3.bias"), shape: weights.shape("fuse.3.weight") };
    this.family = weights.float32("family");
    this.familyShape = weights.shape("family");
    this.bias = weights.float32("bias");
    this.scale = weights.float32("scale")[0];
  }

  /** Logits over families. `ngramIdx`/`ngramSign` and `tokenIds` are already truncated. */
  logits(ngramIdx, ngramSign, tokenIds) {
    const lex = new Float32Array(this.lexDim);
    for (let i = 0; i < ngramIdx.length; i++) {
      this.w.addRow("lex_weight", ngramIdx[i], ngramSign[i], lex);
    }

    const sem = new Float32Array(this.semDim);
    for (const id of tokenIds) this.w.addRow("sem_table", id, 1, sem);
    const denom = Math.max(tokenIds.length, 1);
    for (let d = 0; d < this.semDim; d++) sem[d] /= denom;

    const a = layerNorm(lex, this.lexNormW, this.lexNormB);
    const b = layerNorm(sem, this.semNormW, this.semNormB);
    const x = new Float32Array(a.length + b.length);
    x.set(a, 0);
    x.set(b, a.length);

    const h = linear(x, this.fc1);
    for (let i = 0; i < h.length; i++) h[i] = gelu(h[i]);
    const z = normalize(linear(h, this.fc2));

    const [n, d] = this.familyShape;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let acc = 0;
      for (let j = 0; j < d; j++) acc += this.family[i * d + j] * z[j];
      out[i] = this.scale * acc + this.bias[i];
    }
    return out;
  }
}

function linear(x, { w, b, shape: [out, inp] }) {
  const y = new Float32Array(out);
  for (let i = 0; i < out; i++) {
    let acc = b[i];
    for (let j = 0; j < inp; j++) acc += w[i * inp + j] * x[j];
    y[i] = acc;
  }
  return y;
}

function layerNorm(x, w, b, eps = 1e-5) {
  let mean = 0;
  for (const v of x) mean += v;
  mean /= x.length;
  let variance = 0;
  for (const v of x) variance += (v - mean) ** 2;
  const inv = 1 / Math.sqrt(variance / x.length + eps);
  const y = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) y[i] = (x[i] - mean) * inv * w[i] + b[i];
  return y;
}

function normalize(x) {
  let n = 0;
  for (const v of x) n += v * v;
  n = Math.max(Math.sqrt(n), 1e-12);
  for (let i = 0; i < x.length; i++) x[i] /= n;
  return x;
}

// Exact GELU, 0.5 * x * (1 + erf(x / sqrt(2))).
function gelu(x) {
  return 0.5 * x * (1 + erf(x / Math.SQRT2));
}

// Abramowitz and Stegun 7.1.26, error below 1.5e-7.
function erf(x) {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
  return sign * y;
}
