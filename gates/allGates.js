const { 
  Neuron,
  Network,
  Layer,
  Trainer
} = require("synaptic")

const data = require('./__data__/allGatesData')

const inputLayer = new Layer(7);
const hiddenLayer = new Layer(100);
const outputLayer = new Layer(1);

inputLayer.project(hiddenLayer);
hiddenLayer.project(outputLayer);

const allGatesNetwork = new Network({
  input: inputLayer,
  hidden: [hiddenLayer],
  output: outputLayer
});

const trainer = new Trainer(allGatesNetwork);
trainer.train(data.training, {
    rate: .5,
    iterations: 10000,
    error: .2,
    shuffle: true,
    log: 100,
    cost: Trainer.cost.CROSS_ENTROPY,
});

module.exports = allGatesNetwork