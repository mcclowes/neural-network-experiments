const { 
	Neuron,
	Network,
	Layer,
} = require("synaptic")

const and = {
	type: 'and',
	rules: [
		[[0,0], 0],
		[[0,1], 0],
		[[1,0], 0],
		[[1,1], 1],
	],
}

const nand = {
	type: 'nand',
	rules: [
		[[0,0], 1],
		[[0,1], 1],
		[[1,0], 1],
		[[1,1], 0],
	],
}

const xor = {
	type: 'xor',
	rules: [
		[[0,0], 0],
		[[0,1], 1],
		[[1,0], 1],
		[[1,1], 0],
	],
}

const or = {
	type: 'or',
	rules: [
		[[0,0], 0],
		[[0,1], 1],
		[[1,0], 1],
		[[1,1], 1],
	],
}

const nor = {
	type: 'nor',
	rules: [
		[[0,0], 1],
		[[0,1], 0],
		[[1,0], 0],
		[[1,1], 0],
	],
}

const gates = [and, nand, or, xor, nor]

let twoInputNetwork

gates.map(gate => {
	const inputLayer = new Layer(2);
	const hiddenLayer = new Layer(3);
	const outputLayer = new Layer(1);

	inputLayer.project(hiddenLayer);
	hiddenLayer.project(outputLayer);

	const twoInputNetwork = new Network({
		input: inputLayer,
		hidden: [hiddenLayer],
		output: outputLayer,
	});

	// train the network - learn XOR
	const learningRate = .3;
	
	for (let i = 0; i < 20000; i++) {
		gate.rules.map(rule => {
			twoInputNetwork.activate(rule[0]);
			twoInputNetwork.propagate(learningRate, [rule[1]]);
		})
	}

	// test the network
	console.log(gate.type);
	console.log('[0,0] -> ', Math.round(twoInputNetwork.activate([0,0])));
	console.log('[0,1] -> ', Math.round(twoInputNetwork.activate([0,1])));
	console.log('[1,0] -> ', Math.round(twoInputNetwork.activate([1,0])));
	console.log('[1,1] -> ', Math.round(twoInputNetwork.activate([1,1])));
})