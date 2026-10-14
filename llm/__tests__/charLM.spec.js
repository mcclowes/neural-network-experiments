const { buildVocab, makeDataset, encodeContext, generate } = require("../lm");
const charLM = require("../charLM");
const { corpus } = require("../__data__/corpus");

// Fast, training-free unit tests for the pure toolkit helpers.
describe("lm toolkit", () => {
  const vocab = buildVocab("abc ");

  test("buildVocab is the sorted char set incl. pad", () => {
    expect(vocab.chars).toEqual([" ", "a", "b", "c"]);
    expect(vocab.size).toBe(4);
    expect(vocab.decode(vocab.encode("b"))).toBe("b");
  });

  test("encodeContext left-pads and flattens to contextLength * vocabSize", () => {
    const row = encodeContext(vocab, "c", 2); // -> " c"
    expect(row).toHaveLength(2 * vocab.size);
    expect(row).toEqual([1, 0, 0, 0, /* space */ 0, 0, 0, 1 /* c */]);
  });

  test("makeDataset yields one (context -> next) pair per char", () => {
    const { inputs, outputs } = makeDataset("abc", vocab, 2);
    expect(inputs).toHaveLength(3);
    expect(outputs).toHaveLength(3);
    expect(outputs[0]).toEqual([0, 1, 0, 0]); // first target is 'a'
  });
});

// Training-based tests. Seeded init + full-batch training is deterministic, so
// exact assertions are stable run-to-run — same contract as the gate specs.
describe("char-LM", () => {
  let model, vocab;
  beforeAll(async () => {
    ({ model, vocab } = await charLM.train());
  }, 60000);

  afterAll(() => model && model.dispose());

  charLM.cases.forEach(({ context, next }) => {
    test(`"${context}" -> "${next}"`, () => {
      expect(charLM.predict(model, vocab, context)).toBe(next);
    });
  });

  test("greedy generation reproduces the memorized corpus", () => {
    const seed = corpus.slice(0, charLM.contextLength);
    const text = generate(model, vocab, seed, corpus.length - seed.length, {
      contextLength: charLM.contextLength,
    });
    expect(text.slice(0, corpus.length)).toBe(corpus);
  });

  test("greedy generation is deterministic across trainings", async () => {
    const { model: m2, vocab: v2 } = await charLM.train();
    const seed = corpus.slice(0, charLM.contextLength);
    const opts = { contextLength: charLM.contextLength };
    const a = generate(model, vocab, seed, 60, opts);
    const b = generate(m2, v2, seed, 60, opts);
    m2.dispose();
    expect(b).toBe(a);
  }, 60000);
});
