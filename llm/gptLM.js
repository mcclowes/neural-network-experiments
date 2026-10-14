const fs = require("fs");
const path = require("path");
const { makeGPT } = require("./gpt");

// A char-level GPT: a small transformer decoder (token + positional embeddings,
// causal multi-head self-attention, MLP blocks) trained from scratch on a corpus
// read from a plain text file. Unlike charLM.js's MLP (which sees a bag of
// one-hot context chars), this attends over positions, so it's a genuine (tiny)
// GPT. Drop your own text into __data__/gpt-corpus.txt to train on it; changing
// the file (or any hyperparameter below) invalidates the weight cache.
const text = fs.readFileSync(path.join(__dirname, "__data__", "gpt-corpus.txt"), "utf8");

const gptLM = makeGPT({
  name: "char-GPT",
  text,
  contextLength: 8,
  nEmbd: 32,   // embedding / residual width
  nHead: 4,    // attention heads (headDim = 32/4 = 8)
  nLayer: 2,   // transformer blocks
  learningRate: 0.01,
  epochs: 500,
  // Trained weights are cached here so `npm run gpt` only trains once; the cache
  // is keyed on the corpus + hyperparameters and rebuilt when either changes.
  cachePath: path.join(__dirname, "__cache__", "char-gpt.json"),
});

if (require.main === module) gptLM.run();

module.exports = gptLM;
