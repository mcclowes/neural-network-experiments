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
paint-an-image.js   standalone browser perceptron sketch (unused)
```

## Notes

- `synaptic` remains a dependency only because `tic-tac-toe` still uses it.
  Porting tic-tac-toe to TensorFlow.js would let it be removed entirely.
- `mnist` is listed as a dependency but is only referenced in commented-out code.
