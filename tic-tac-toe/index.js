const { 
  Neuron,
  Network,
  Layer,
  Trainer
} = require("synaptic")

const data = require('./__data__/data')

const inputLayer = new Layer(18);
const hiddenLayer = new Layer(100);
const outputLayer = new Layer(9);

inputLayer.project(hiddenLayer);
hiddenLayer.project(outputLayer);

const ticTacToeNetwork = new Network({
  input: inputLayer,
  hidden: [hiddenLayer],
  output: outputLayer
});

// const mnist = require('mnist'); 
// const set = mnist.set(700, 20);

const trainingSet = data.training;

const trainer = new Trainer(ticTacToeNetwork);
trainer.train(trainingSet, {
    rate: .5,
    iterations: 10000,
    error: .1,
    shuffle: true,
    log: 10,
    cost: Trainer.cost.CROSS_ENTROPY
});

module.exports = ticTacToeNetwork