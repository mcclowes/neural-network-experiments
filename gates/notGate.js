const { 
	Neuron,
	Network,
	Layer,
} = require("synaptic")

const data = require('./__data__/allGatesData')

const inputLayer = new Layer(1); // number of nodes
const outputLayer = new Layer(1);

inputLayer.project(outputLayer);

const oneInputNetwork = new Network({
	input: inputLayer,
	output: outputLayer,
});

// train the network - learn XOR
var learningRate = .3;

for (var i = 0; i < 20000; i++) {
	data.gates.not.rules.map(rule => {
		oneInputNetwork.activate(rule.input);
		oneInputNetwork.propagate(learningRate, [rule.output]);
	})
}

module.exports = oneInputNetwork