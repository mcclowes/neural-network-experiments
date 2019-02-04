const { 
	Neuron,
	Network,
	Layer,
} = require("synaptic")

const inputLayer = new Layer(1); // number of nodes
const outputLayer = new Layer(1);

inputLayer.project(outputLayer);

const oneInputNetwork = new Network({
	input: inputLayer,
	output: outputLayer,
});

const not = {
	type: 'not',
	rules: [
		[[0],1],
		[[1],0],
	],
}


// train the network - learn XOR
var learningRate = .3;

for (var i = 0; i < 20000; i++) {
	not.map(rule => {
		oneInputNetwork.activate(rule[0]);
		oneInputNetwork.propagate(learningRate, [rule[1]]);
	})
}

console.log(oneInputNetwork)
// test the network
console.log(Math.round(oneInputNetwork.activate([0])));
console.log(Math.round(oneInputNetwork.activate([1])));