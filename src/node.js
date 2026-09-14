import { readFile } from "node:fs/promises";
import { Iconic } from "./index.js";

export * from "./index.js";

/** Loads the model files that ship inside this package. No network needed. */
export async function loadBundled() {
  const dir = new URL("../model/", import.meta.url);
  const [manifest, weights, tokenizer] = await Promise.all([
    readFile(new URL("js/manifest.json", dir), "utf8").then(JSON.parse),
    readFile(new URL("js/weights.bin", dir)),
    readFile(new URL("tokenizer.json", dir), "utf8").then(JSON.parse),
  ]);
  const buffer = weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength);
  return new Iconic({ manifest, weights: buffer, tokenizer });
}
