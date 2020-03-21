const { 
  Neuron,
  Network,
  Layer,
  Trainer
} = require("synaptic")

const data = require('./data')

const inputLayer = new Layer(9);
const hiddenLayer = new Layer(30);
const outputLayer = new Layer(9);

inputLayer.project(hiddenLayer);
hiddenLayer.project(outputLayer);

const myNetwork = new Network({
  input: inputLayer,
  hidden: [hiddenLayer],
  output: outputLayer
});

// const mnist = require('mnist'); 
// const set = mnist.set(700, 20);

const trainingSet = data.training;

const trainer = new Trainer(myNetwork);
trainer.train(trainingSet, {
    rate: .2,
    iterations: 100000,
    error: .1,
    shuffle: true,
    log: 1,
    cost: Trainer.cost.CROSS_ENTROPY
});

trainingSet.map((_,i) => {
  console.log('>>> --------------------------------')
  const result = myNetwork.activate(trainingSet[i].input)
  console.log(result);
  console.log(trainingSet[i].output);
  console.log(trainingSet[i].output.map((x,j) => x - result[j]))
})