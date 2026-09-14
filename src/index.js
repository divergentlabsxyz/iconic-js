import { hashFeatures } from "./hash.js";
import { Network, Weights } from "./model.js";
import { Tokenizer } from "./tokenizer.js";

export { Tokenizer } from "./tokenizer.js";
export { features, hashFeatures } from "./hash.js";

/** Where the model files are published. */
export const HUGGING_FACE_URL = "https://huggingface.co/amsintelligence/iconic/resolve/main";

export class Iconic {
  /**
   * @param {object} files
   * @param {object} files.manifest  parsed js/manifest.json
   * @param {ArrayBuffer} files.weights  js/weights.bin
   * @param {object} files.tokenizer  parsed tokenizer.json
   */
  constructor({ manifest, weights, tokenizer }) {
    this.manifest = manifest;
    this.families = manifest.families;
    this.defaults = manifest.default_member;
    this.buckets = manifest.buckets;
    this.maxNgrams = manifest.max_ngrams;
    this.maxTokens = manifest.max_tokens;
    this.tokenizer = new Tokenizer(tokenizer);
    this.network = new Network(new Weights(manifest, weights));
  }

  /**
   * Loads the model files from a URL with the Hugging Face repo layout:
   * `js/manifest.json`, `js/weights.bin` and `tokenizer.json`.
   */
  static async load(baseUrl = HUGGING_FACE_URL, init) {
    const get = async (path) => {
      const res = await fetch(`${baseUrl.replace(/\/$/, "")}/${path}`, init);
      if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
      return res;
    };
    const [manifest, weights, tokenizer] = await Promise.all([
      get("js/manifest.json").then((r) => r.json()),
      get("js/weights.bin").then((r) => r.arrayBuffer()),
      get("tokenizer.json").then((r) => r.json()),
    ]);
    return new Iconic({ manifest, weights, tokenizer });
  }

  /** The model inputs for `text`, as the Core ML model receives them. */
  inputs(text) {
    const { idx, sign } = hashFeatures(text, { buckets: this.buckets });
    let tokenIds = this.tokenizer.encode(text).slice(0, this.maxTokens);
    if (tokenIds.length === 0) tokenIds = [this.tokenizer.unkId];
    return {
      ngramIdx: idx.subarray(0, this.maxNgrams),
      ngramSign: sign.subarray(0, this.maxNgrams),
      tokenIds,
    };
  }

  /** Raw logits, one per family, in `families` order. */
  logits(text) {
    const { ngramIdx, ngramSign, tokenIds } = this.inputs(text);
    return this.network.logits(ngramIdx, ngramSign, tokenIds);
  }

  /**
   * The best-matching icons for `text`.
   * @returns {{family: string, icon: string, score: number}[]}
   */
  predict(text, { k = 3 } = {}) {
    const logits = this.logits(text);
    const order = Array.from(logits.keys())
      .sort((a, b) => logits[b] - logits[a])
      .slice(0, k);
    const max = logits[order[0]];
    let z = 0;
    for (const v of logits) z += Math.exp(v - max);
    return order.map((i) => ({
      family: this.families[i],
      icon: this.defaults[this.families[i]] ?? this.families[i],
      score: Math.exp(logits[i] - max) / z,
    }));
  }
}
