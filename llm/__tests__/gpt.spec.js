const fs = require("fs");
const os = require("os");
const path = require("path");
const { tf } = require("../../gates/tf");
const { buildVocab } = require("../lm");
const {
  encodeTokens,
  makeBlocks,
  causalMask,
  initParams,
  forward,
  disposeParams,
  generate,
  makeGPT,
  signatureOf,
  saveWeights,
  loadWeights,
} = require("../gpt");
const gptLM = require("../gptLM");

// Fast, training-free unit tests for the pure toolkit helpers.
describe("gpt toolkit", () => {
  const vocab = buildVocab("abc "); // sorted: [" ", "a", "b", "c"]

  test("encodeTokens maps chars to sorted-vocab ids and round-trips", () => {
    expect(encodeTokens(vocab, "cab")).toEqual([3, 1, 2]);
    expect(encodeTokens(vocab, "cab").map(vocab.decode).join("")).toBe("cab");
  });

  test("makeBlocks yields (len - contextLength) windows shifted by one", () => {
    const { xs, ys } = makeBlocks(encodeTokens(vocab, "abcb"), 2); // ids: [1,2,3,2]
    expect(xs).toEqual([[1, 2], [2, 3]]);
    expect(ys).toEqual([[2, 3], [3, 2]]); // each target is its window shifted left
  });

  test("causalMask blocks future positions and allows past + self", () => {
    const T = 4;
    const mask = causalMask(T).arraySync();
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < T; j++) {
        if (j > i) expect(mask[i][j]).toBeLessThan(0); // future -> ~ -inf
        else expect(mask[i][j]).toBeCloseTo(0); // past + self -> unmasked (may be -0)
      }
    }
  });

  test("forward returns logits shaped [batch, contextLength, vocabSize]", () => {
    const config = { contextLength: 3, nEmbd: 8, nHead: 2, nLayer: 1, seed: 1 };
    const params = initParams(vocab, config);
    const idx = tf.tensor2d([[0, 1, 2], [2, 1, 0]], [2, 3], "int32");
    const logits = forward(params, idx, config);
    expect(logits.shape).toEqual([2, 3, vocab.size]);
    idx.dispose();
    logits.dispose();
    disposeParams(params);
  });

  test("signatureOf changes with the corpus and the hyperparameters", () => {
    const cfg = { contextLength: 8, nEmbd: 32, nHead: 4, nLayer: 2, learningRate: 0.01, epochs: 500, seed: 42 };
    expect(signatureOf("hello", cfg)).toBe(signatureOf("hello", cfg));
    expect(signatureOf("hello", cfg)).not.toBe(signatureOf("world", cfg));
    expect(signatureOf("hello", cfg)).not.toBe(signatureOf("hello", { ...cfg, seed: 43 }));
  });
});

// A tiny corpus of unique 8-grams is fully memorizable, so — with seeded,
// deterministic training — greedy generation reproduces it exactly. This is the
// GPT analog of a gate truth table, the same contract as the char-LM spec.
describe("char-GPT memorizes a tiny corpus", () => {
  const pangram = "the quick brown fox jumps over the lazy dog. ";
  const tinyGPT = makeGPT({
    name: "tiny-GPT",
    text: pangram,
    contextLength: 8,
    nEmbd: 32,
    nHead: 4,
    nLayer: 2,
    learningRate: 0.01,
    epochs: 500,
    cases: [
      { context: "the quic", next: "k" },
      { context: "brown fo", next: "x" },
      { context: "umps ove", next: "r" },
      { context: "he lazy ", next: "d" },
    ],
  });

  let params, vocab;
  beforeAll(async () => {
    ({ params, vocab } = await tinyGPT.train());
  }, 60000);

  afterAll(() => params && disposeParams(params));

  tinyGPT.cases.forEach(({ context, next }) => {
    test(`"${context}" -> "${next}"`, () => {
      expect(tinyGPT.predict(params, vocab, context)).toBe(next);
    });
  });

  test("greedy generation reproduces the memorized corpus", () => {
    const seed = pangram.slice(0, tinyGPT.contextLength);
    const text = generate(params, vocab, seed, pangram.length - seed.length, tinyGPT);
    expect(text.slice(0, pangram.length)).toBe(pangram);
  });

  test("greedy generation is deterministic across trainings", async () => {
    const { params: p2, vocab: v2 } = await tinyGPT.train();
    const seed = pangram.slice(0, tinyGPT.contextLength);
    const a = generate(params, vocab, seed, 40, tinyGPT);
    const b = generate(p2, v2, seed, 40, tinyGPT);
    disposeParams(p2);
    expect(b).toBe(a);
  }, 60000);
});

// The shipped experiment trains on a larger file corpus with repeated contexts,
// so it can't memorize verbatim — it learns. We assert learning (loss drops) and
// well-formed output rather than exact text.
describe("char-GPT on the file corpus", () => {
  let params, vocab, history;
  beforeAll(async () => {
    ({ params, vocab, history } = await gptLM.train());
  }, 60000);

  afterAll(() => params && disposeParams(params));

  test("training substantially lowers the loss", () => {
    expect(history[history.length - 1]).toBeLessThan(history[0] * 0.5);
    expect(history[history.length - 1]).toBeLessThan(1.0);
  });

  test("generated text is the requested length and stays in-vocab", () => {
    const seed = gptLM.text.slice(0, gptLM.contextLength);
    const out = generate(params, vocab, seed, 50, gptLM);
    expect(out).toHaveLength(seed.length + 50);
    expect([...out].every((c) => vocab.chars.includes(c))).toBe(true);
  });
});

// Weights are a deterministic function of (corpus, config), so a cache keyed on
// their signature is sound: reloading must reproduce generation byte-for-byte,
// and a changed corpus/config must miss.
describe("weight cache", () => {
  const cachePath = path.join(os.tmpdir(), "nn-gpt-cache-test.json");
  let params, vocab;

  beforeAll(async () => {
    ({ params, vocab } = await gptLM.train());
    saveWeights(params, gptLM.text, gptLM, cachePath);
  }, 60000);

  afterAll(() => {
    if (params) disposeParams(params);
    if (fs.existsSync(cachePath)) fs.unlinkSync(cachePath);
  });

  test("loaded weights reproduce generation byte-for-byte (no retrain)", () => {
    const loaded = loadWeights(gptLM.text, gptLM, cachePath);
    const seed = gptLM.text.slice(0, gptLM.contextLength);
    const a = generate(params, vocab, seed, 60, gptLM);
    const b = generate(loaded.params, loaded.vocab, seed, 60, gptLM);
    disposeParams(loaded.params);
    expect(b).toBe(a);
  });

  test("a stale signature (changed corpus or config) misses the cache", () => {
    expect(loadWeights("some other corpus", gptLM, cachePath)).toBeNull();
    expect(loadWeights(gptLM.text, { ...gptLM, seed: 999 }, cachePath)).toBeNull();
  });

  test("a missing cache file returns null", () => {
    expect(loadWeights(gptLM.text, gptLM, path.join(os.tmpdir(), "nn-gpt-nope.json"))).toBeNull();
  });
});
