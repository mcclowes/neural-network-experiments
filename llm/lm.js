// Shared toolkit for the char-level language-model experiment. A "simple LLM":
// predict the next character from the previous N characters, then generate text
// by feeding predictions back in. Same design as gates/tf.js — the mechanics
// live here, the corpus + hyperparameters live in the experiment file.
//
// Reuses gates/tf.js purely for its Node-23 compat shim + shared tf instance.
const { tf, DEFAULT_SEED } = require("../gates/tf");

const PAD = " "; // left-pads short contexts (corpora here are prose, so space is safe)

// Vocab is just the sorted set of characters in the corpus, so encoding is
// deterministic run-to-run.
const buildVocab = (text) => {
  const chars = [...new Set([PAD, ...text])].sort();
  const stoi = new Map(chars.map((c, i) => [c, i]));
  return {
    chars,
    size: chars.length,
    encode: (ch) => stoi.get(ch),
    decode: (i) => chars[i],
  };
};

// One-hot a single character into a vocab-sized array.
const oneHot = (vocab, ch) => {
  const v = new Array(vocab.size).fill(0);
  v[vocab.encode(ch)] = 1;
  return v;
};

// Left-pad/truncate a context to exactly contextLength chars.
const fitContext = (context, contextLength) =>
  context.slice(-contextLength).padStart(contextLength, PAD);

// Flatten a context into a single (contextLength * vocabSize) one-hot input row.
const encodeContext = (vocab, context, contextLength) =>
  [...fitContext(context, contextLength)].flatMap((ch) => oneHot(vocab, ch));

// Build (context -> next char) training pairs by sliding a window over the text.
const makeDataset = (text, vocab, contextLength) => {
  const inputs = [];
  const outputs = [];
  for (let i = 0; i < text.length; i++) {
    const context = text.slice(Math.max(0, i - contextLength), i);
    inputs.push(encodeContext(vocab, context, contextLength));
    outputs.push(oneHot(vocab, text[i]));
  }
  return { inputs, outputs };
};

// Seeded init + full-batch training => deterministic, so tests assert exact text.
const buildModel = (vocab, contextLength, { hidden = 64, learningRate = 0.1, seed = DEFAULT_SEED } = {}) => {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: hidden,
      activation: "relu",
      inputShape: [contextLength * vocab.size],
      kernelInitializer: tf.initializers.glorotUniform({ seed }),
    })
  );
  model.add(
    tf.layers.dense({
      units: vocab.size,
      activation: "softmax",
      kernelInitializer: tf.initializers.glorotUniform({ seed: seed + 1 }),
    })
  );
  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
};

const trainLM = async (text, { contextLength, hidden, learningRate, epochs = 400, seed } = {}) => {
  const vocab = buildVocab(text);
  const { inputs, outputs } = makeDataset(text, vocab, contextLength);
  const xs = tf.tensor2d(inputs);
  const ys = tf.tensor2d(outputs);
  const model = buildModel(vocab, contextLength, { hidden, learningRate, seed });
  const history = await model.fit(xs, ys, { epochs, shuffle: false, verbose: 0 });
  xs.dispose();
  ys.dispose();
  return { model, vocab, history };
};

// Greedy (argmax) next-char prediction — deterministic, so it is what tests use.
const predictNext = (model, vocab, context, contextLength) =>
  tf.tidy(() => {
    const input = tf.tensor2d([encodeContext(vocab, context, contextLength)]);
    return vocab.decode(model.predict(input).argMax(1).dataSync()[0]);
  });

// Generate `length` chars by feeding each prediction back in. Greedy by default
// (reproducible); temperature > 0 enables random sampling for the demo.
const generate = (model, vocab, seed, length, { contextLength, temperature = 0 } = {}) => {
  let out = seed;
  for (let i = 0; i < length; i++) {
    const next =
      temperature > 0
        ? sample(model, vocab, out, contextLength, temperature)
        : predictNext(model, vocab, out, contextLength);
    out += next;
  }
  return out;
};

const sample = (model, vocab, context, contextLength, temperature) =>
  tf.tidy(() => {
    const input = tf.tensor2d([encodeContext(vocab, context, contextLength)]);
    const logits = model.predict(input).log().div(temperature);
    const idx = tf.multinomial(logits, 1).dataSync()[0];
    return vocab.decode(idx);
  });

// Builds a char-LM experiment from pure config, mirroring gates' makeExperiment.
// `cases` is [{ context, next }] — the next-char analog of a gate truth table.
const makeCharLM = ({ name, text, contextLength, hidden, learningRate, epochs, cases = [], seed }) => {
  const train = (opts = {}) =>
    trainLM(text, { contextLength, hidden, learningRate, epochs, seed, ...opts });

  const predict = (model, vocab, context) => predictNext(model, vocab, context, contextLength);

  const run = async () => {
    const { model, vocab } = await train();
    console.log(`${name} — vocab ${vocab.size}, context ${contextLength}, ${text.length} chars\n`);
    const seed = text.slice(0, contextLength);
    console.log("greedy:", JSON.stringify(generate(model, vocab, seed, 120, { contextLength })));
    console.log("temp0.7:", JSON.stringify(generate(model, vocab, seed, 120, { contextLength, temperature: 0.7 })));
    model.dispose();
  };

  return { name, text, contextLength, cases, train, predict, generate, run };
};

module.exports = {
  buildVocab,
  makeDataset,
  encodeContext,
  buildModel,
  trainLM,
  predictNext,
  generate,
  makeCharLM,
};
