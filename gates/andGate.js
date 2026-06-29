const { makeExperiment } = require("./tf");
const data = require("./__data__/allGatesData");

// Clean 2-input AND. (The original synaptic version padded inputs to varying
// widths — experimental cruft, dropped here.)
const andGate = makeExperiment({
  name: "AND",
  cases: data.gates.and.rules, // [{ input: [0,0], output: 0 }, ...]
  inputShape: [2],
  layers: [
    { units: 2, activation: "relu" },
    { units: 1, activation: "sigmoid" },
  ],
  learningRate: 0.1,
  epochs: 400,
});

if (require.main === module) andGate.run();

module.exports = andGate;
