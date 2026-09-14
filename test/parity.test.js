// Checks the SDK against outputs produced by the Python reference code.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { features, hashFeatures } from "../src/hash.js";
import { loadBundled } from "../src/node.js";

const golden = JSON.parse(readFileSync(new URL("fixtures/golden.json", import.meta.url), "utf8"));
const iconic = await loadBundled();

test("normalizer matches", () => {
  for (const g of golden) {
    assert.equal(iconic.tokenizer.normalize(g.text), g.normalized, JSON.stringify(g.text));
  }
});

test("tokens match", () => {
  for (const g of golden) {
    assert.deepEqual(iconic.tokenizer.tokenize(g.text), g.tokens, JSON.stringify(g.text));
    assert.deepEqual(iconic.tokenizer.encode(g.text), g.token_ids, JSON.stringify(g.text));
  }
});

test("n-gram features match", () => {
  for (const g of golden) {
    assert.deepEqual(features(g.text).slice(0, 512), g.features, JSON.stringify(g.text));
  }
});

test("hashes match", () => {
  for (const g of golden) {
    const { ngramIdx, ngramSign } = iconic.inputs(g.text);
    assert.deepEqual(Array.from(ngramIdx), g.ngram_idx, JSON.stringify(g.text));
    ngramSign.forEach((s, i) => assert.ok(Math.abs(s - g.ngram_sign[i]) < 1e-6));
  }
  assert.equal(hashFeatures("", { buckets: 65536 }).count, 0);
});

test("predictions agree with the float model", () => {
  let top1 = 0;
  let inTop3 = 0;
  for (const g of golden) {
    const [best] = iconic.predict(g.text, { k: 1 });
    if (best.family === g.top5[0]) top1++;
    if (g.top5.slice(0, 3).includes(best.family)) inTop3++;
  }
  // The bundle is 4-bit quantised, so a few near-ties may flip.
  assert.ok(top1 / golden.length >= 0.9, `top-1 agreement ${top1}/${golden.length}`);
  assert.ok(inTop3 / golden.length >= 0.97, `top-3 agreement ${inTop3}/${golden.length}`);
});

test("predict returns ranked icons", () => {
  const out = iconic.predict("buy milk");
  assert.equal(out.length, 3);
  assert.ok(out[0].score >= out[1].score && out[1].score >= out[2].score);
  assert.equal(typeof out[0].icon, "string");
});
