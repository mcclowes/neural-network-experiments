const { 
  Neuron,
  Network,
  Layer,
  Trainer
} = require("synaptic")

const data = require('./__data__/allGatesData')

const inputLayer = new Layer(3);
const hiddenLayer = new Layer(10);
const outputLayer = new Layer(1);

inputLayer.project(hiddenLayer);
hiddenLayer.project(outputLayer);

const andGateNetwork = new Network({
  input: inputLayer,
  hidden: [hiddenLayer],
  output: outputLayer
});

const trainingSet = [
  {
    input: [1, data.gates.and.rules[0].input[0], data.gates.and.rules[0].input[1]],
    output: [data.gates.and.rules[0].output],
    description: "AND - [0, 0] -> 0",
  },
  {
    input: [1, 0, 0, 0, 0, data.gates.and.rules[1].input[0], data.gates.and.rules[1].input[1]],
    output: [data.gates.and.rules[1].output],
    description: "AND - [0, 1] -> 0",
  },
  {
    input: [1, 0, 0, 0, 0, data.gates.and.rules[2].input[0], data.gates.and.rules[2].input[1]],
    output: [data.gates.and.rules[2].output],
    description: "AND - [1, 0] -> 0",
  },
  {
    input: [1, data.gates.and.rules[3].input[0], data.gates.and.rules[3].input[1]],
    output: [data.gates.and.rules[3].output],
    description: "AND - [1, 1] -> 1",
  },
]

const trainer = new Trainer(andGateNetwork);
trainer.train(trainingSet, {
    rate: .5,
    iterations: 1000,
    error: .2,
    shuffle: true,
    log: 100,
    cost: Trainer.cost.CROSS_ENTROPY
});

module.exports = andGateNetwork