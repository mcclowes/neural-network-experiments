const { makeExperiment } = require("./tf");

// AND, NAND, OR, XOR, NOR — each trained on its own 2 -> 4 -> 1 net.
const truthTables = {
  AND: [[[0, 0], 0], [[0, 1], 0], [[1, 0], 0], [[1, 1], 1]],
  NAND: [[[0, 0], 1], [[0, 1], 1], [[1, 0], 1], [[1, 1], 0]],
  OR: [[[0, 0], 0], [[0, 1], 1], [[1, 0], 1], [[1, 1], 1]],
  XOR: [[[0, 0], 0], [[0, 1], 1], [[1, 0], 1], [[1, 1], 0]],
  NOR: [[[0, 0], 1], [[0, 1], 0], [[1, 0], 0], [[1, 1], 0]],
};

const experiments = Object.entries(truthTables).map(([name, table]) =>
  makeExperiment({
    name,
    cases: table.map(([input, output]) => ({ input, output })),
    inputShape: [2],
    layers: [
      { units: 4, activation: "relu" },
      { units: 1, activation: "sigmoid" },
    ],
    learningRate: 0.1,
    epochs: 500,
  })
);

if (require.main === module) {
  (async () => {
    for (const experiment of experiments) await experiment.run();
  })();
}

module.exports = experiments;
