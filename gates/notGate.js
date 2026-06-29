const { makeExperiment } = require("./tf");
const data = require("./__data__/allGatesData");

// NOT is linearly separable, so a single sigmoid neuron suffices.
const notGate = makeExperiment({
  name: "NOT",
  cases: data.gates.not.rules, // [{ input: [0], output: 1 }, ...]
  inputShape: [1],
  layers: [{ units: 1, activation: "sigmoid" }],
  learningRate: 0.1,
  epochs: 300,
});

if (require.main === module) notGate.run();

module.exports = notGate;
