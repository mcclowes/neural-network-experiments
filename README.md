# neural-network-experiments

Small, self-contained experiments learning neural networks with [`synaptic`](https://github.com/cazala/synaptic) in plain Node.js. Each experiment trains a network from scratch on toy data and is runnable via an npm script.

## Setup

```bash
npm install
```

## Experiments

| Script | File | What it does |
|---|---|---|
| `npm run gates:notGate` | `gates/notGate.js` | Single input→output network learning NOT |
| `npm run gates:andGate` | `gates/andGate.js` | 3-input network learning AND (padded input vectors) |
| `npm run gates:twoInputGates` | `gates/twoInputGates.js` | Trains AND, NAND, OR, XOR, NOR each on a 2-3-1 net; logs predictions |
| `npm run gates:allGates` | `gates/allGates.js` | One 7-100-1 network trained on all gates at once |
| `npm run tic-tac-toe` | `tic-tac-toe/index.js` | 18-100-9 network learning tic-tac-toe moves from board state |

Most experiment files `module.exports` the trained network; running the script trains it (and, for `twoInputGates`, prints results).

`paint-an-image.js` is an older AngularJS-era browser sketch (`Architect.Perceptron`) and is **not** wired to an npm script or used in tests.

## Tests

```bash
npm test   # jest
```

Tests live in `__tests__/` folders and assert trained networks classify their training examples correctly.

> ⚠️ `gates/__tests__/allGates.spec.js` is **flaky**: the shared all-gates network trains only to `error: .2`, so a rounded prediction occasionally lands on the wrong side and one assertion fails. Re-running usually passes. Tightening the training `error` / increasing `iterations` in `gates/allGates.js` would stabilise it.

## Project layout

```
gates/            gate-learning experiments
  __data__/       training data (allGatesData.js)
  __tests__/      jest specs
tic-tac-toe/
  __data__/       board-state training data
  __tests__/
paint-an-image.js standalone browser perceptron sketch (unused)
```

## Notes

- `mnist` is listed as a dependency but is currently only referenced in commented-out code.
