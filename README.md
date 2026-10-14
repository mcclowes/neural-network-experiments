# neural-network-experiments

Small, self-contained experiments learning neural networks in plain Node.js.
The gate experiments use [TensorFlow.js](https://www.npmjs.com/package/@tensorflow/tfjs-node);
`tic-tac-toe` still uses the older [`synaptic`](https://github.com/cazala/synaptic)
(not yet ported). Each experiment trains a network from scratch on toy data and
is runnable via an npm script.

## Setup

```bash
npm install
```

## Experiments

| Script | File | What it does |
|---|---|---|
| `npm run gates:notGate` | `gates/notGate.js` | NOT — single sigmoid neuron (1-1) |
| `npm run gates:andGate` | `gates/andGate.js` | AND — 2-2-1 net |
| `npm run gates:twoInputGates` | `gates/twoInputGates.js` | AND, NAND, OR, XOR, NOR — each on its own 2-4-1 net |
| `npm run gates:allGates` | `gates/allGates.js` | All five gates in one 7-16-8-1 net (one-hot gate selector + 2 operands) |
| `npm run tic-tac-toe` | `tic-tac-toe/index.js` | 18-100-9 network learning tic-tac-toe moves (synaptic) |
| `npm run llm` | `llm/charLM.js` | Char-level language model — MLP predicting the next char, generates text |
| `npm run gpt` | `llm/gptLM.js` | Char-level **GPT** — transformer (causal self-attention) predicting the next char, generates text |

Each gate file `module.exports` an experiment object (or array, for
`twoInputGates`) with `train(opts)` / `predict(model, input)` / `cases`; running
the script trains the net and prints its predictions.

`paint-an-image.js` is an older AngularJS-era browser sketch (`Architect.Perceptron`)
and is **not** wired to an npm script or used in tests.

## TensorFlow.js gate toolkit

The gate experiments share `gates/tf.js`, which provides:

- **A compatibility shim** — tfjs-node 4.22 calls `util.isNullOrUndefined`, which
  Node **removed in v23**, so it crashes on Node 23+ without the shim. Remove it
  once tfjs-node ships a fix ([tfjs#8266](https://github.com/tensorflow/tfjs/issues/8266)).
- **`makeExperiment(config)`** — builds a binary-classifier experiment from pure
  config (layer sizes, learning rate, epochs, cases). Each experiment file is
  just that config; the mechanics (build / train / predict / log) live here.
- **Deterministic training** — seeded weight init + full-batch training (datasets
  are ≤20 rows) means results are reproducible, so tests assert exact outputs
  with no retry loop.

## Char-level language model (`llm/`)

A "simple LLM": a char-level MLP trained from scratch that predicts the next
character from the previous N characters, then generates text by feeding its
predictions back in. Same design as the gate toolkit — mechanics in `llm/lm.js`,
corpus + hyperparameters in `llm/charLM.js`.

- **`buildVocab` / `makeDataset`** — turn a corpus into `(context -> next char)`
  one-hot training pairs by sliding a window over the text.
- **Deterministic training** — seeded init + full-batch training on a tiny,
  memorizable corpus means greedy generation reproduces the corpus exactly, so
  the spec asserts exact text (the LM analog of a gate truth table).
- **`generate`** — greedy (argmax) by default for reproducibility; pass
  `temperature > 0` for random sampling. Past the training data, greedy decoding
  falls into the usual short repeating loop — an honest char-LM artifact.

## Char-level GPT (`llm/gpt.js`)

The transformer sequel to the char-LM. Where `charLM.js` feeds a *bag* of one-hot
context chars into an MLP, `gptLM.js` is a real (tiny) **GPT**: token + positional
embeddings, causal multi-head self-attention, and MLP blocks with residuals +
LayerNorm — so it attends over positions instead of ignoring their order.

`tf.layers` has no causal self-attention layer, so unlike `lm.js` (which leans on
`tf.sequential`) the transformer is built from low-level tf ops over `tf.variable`
params and trained with a hand-rolled autodiff loop (`optimizer.minimize`):

- **`makeBlocks`** — slides length-`contextLength` windows over the corpus, each
  paired with its next-token (shift-by-one) targets. The standard GPT objective:
  predict the next char at *every* position, not just the last.
- **`forward`** — pre-norm transformer: `causalMask` (additive `-1e9` above the
  diagonal) enforces that position *t* only attends to ≤ *t*, which is what makes
  it a *causal* LM rather than a bidirectional encoder.
- **Deterministic training** — seeded init (`tf.randomNormal` with per-tensor
  seeds) + full-batch Adam make training reproducible run-to-run, so the spec can
  assert exact generated text — same contract as the char-LM and gate specs.

### Corpus

`gptLM.js` reads its training text from **`llm/__data__/gpt-corpus.txt`** — drop
in your own text there to train on it (a few KB trains in seconds; much larger
would want mini-batching, which this full-batch trainer doesn't do yet). The
shipped corpus is a handful of sentences with repeated substrings, so — unlike
the char-LM's unique-8-gram pangram — the model can't memorize it verbatim; it
learns to recombine phrases. The spec reflects this: a *tiny* pangram is still
asserted to reproduce exactly (memorization, the truth-table contract), while the
file corpus is only asserted to lower its loss and emit well-formed text.

### Weight cache

Training is deterministic, so trained weights are a pure function of the corpus +
hyperparameters. `gptLM.js` therefore caches them to `llm/__cache__/` (gitignored,
keyed on a signature over exactly those inputs): the first `npm run gpt` trains
and saves; later runs reload in a fraction of a second. Editing the corpus or any
hyperparameter changes the signature and transparently retrains. Only `run()`
uses the cache — `train()` always trains fresh, so tests stay honest.

## Tests

```bash
npm test   # jest
```

Specs live in `gates/__tests__/` and drive each experiment through the shared
`gates/test-helpers.js` harness, asserting every truth-table case. Training is
deterministic, so the suite is stable run-to-run.

## Project layout

```
gates/
  tf.js             tfjs loader: compat shim + shared experiment toolkit
  test-helpers.js   jest harness shared by the gate specs
  notGate.js andGate.js twoInputGates.js allGates.js   experiment configs
  __data__/         training data (allGatesData.js)
  __tests__/        jest specs
tic-tac-toe/        18-100-9 net (still synaptic; unported)
  __data__/         board-state training data
  __tests__/
llm/
  lm.js             char-LM (MLP) toolkit: vocab / dataset / build / train / generate
  charLM.js         char-LM experiment config (corpus + hyperparameters) + run
  gpt.js            char-GPT toolkit: transformer forward + autodiff train + weight cache + generate
  gptLM.js          char-GPT experiment config (corpus path + hyperparameters) + run
  __data__/         corpus.js (char-LM text) + gpt-corpus.txt (char-GPT text)
  __cache__/        cached char-GPT weights (gitignored, regenerated on demand)
  __tests__/        jest specs (toolkit units + memorization + learning + cache)
paint-an-image.js   standalone browser perceptron sketch (unused)
```

## Notes

- `synaptic` remains a dependency only because `tic-tac-toe` still uses it.
  Porting tic-tac-toe to TensorFlow.js would let it be removed entirely.
- `mnist` is listed as a dependency but is only referenced in commented-out code.
