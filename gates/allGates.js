const { makeExperiment } = require("./tf");
const data = require("./__data__/allGatesData");

// One network learns every gate at once. Each input is a 7-vector:
// [5 one-hot gate selector | 2 operands]; output is the gate result.
const allGates = makeExperiment({
  name: "ALL GATES",
  cases: data.training.map((t) => ({
    input: t.input,
    output: t.output[0],
    label: t.description,
  })),
  inputShape: [7],
  layers: [
    { units: 16, activation: "relu" },
    { units: 8, activation: "relu" },
    { units: 1, activation: "sigmoid" },
  ],
  learningRate: 0.05,
  epochs: 800,
});

if (require.main === module) allGates.run();

module.exports = allGates;
