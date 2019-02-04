const { 
	Neuron,
	Network,
	Layer,
} = require("synaptic")

const gateCodes = {
	AND: 1,
	NAND: 2,
	XOR: 3,
	OR: 4,
	NOR: 5,
	NOT: 6,
}

const and = {
	type: 'AND',
	rules: [
		[[0,0], 0],
		[[0,1], 0],
		[[1,0], 0],
		[[1,1], 1],
	],
}

const nand = {
	type: 'NAND',
	rules: [
		[[0,0], 1],
		[[0,1], 1],
		[[1,0], 1],
		[[1,1], 0],
	],
}

const xor = {
	type: 'XOR',
	rules: [
		[[0,0], 0],
		[[0,1], 1],
		[[1,0], 1],
		[[1,1], 0],
	],
}

const or = {
	type: 'OR',
	rules: [
		[[0,0], 0],
		[[0,1], 1],
		[[1,0], 1],
		[[1,1], 1],
	],
}

const nor = {
	type: 'NOR',
	rules: [
		[[0,0], 1],
		[[0,1], 0],
		[[1,0], 0],
		[[1,1], 0],
	],
}

// const not = {
// 	type: 'NOT',
// 	rules: [
// 		[[0],1],
// 		[[1],0],
// 	],
// }

const gates = [and, nand, or, xor, nor,]

const inputLayer = new Layer(3);
const hiddenLayer1 = new Layer(15);
const hiddenLayer2 = new Layer(15);
const outputLayer = new Layer(1);

inputLayer.project(hiddenLayer1);
hiddenLayer1.project(hiddenLayer2);
hiddenLayer2.project(outputLayer);

const allGatesNetwork = new Network({
	input: inputLayer,
	hidden: [hiddenLayer1, hiddenLayer2],
	output: outputLayer,
});

gates.map(gate => {
	const learningRate = .3;

	for (let i = 0; i < 50000; i++) {
		gate.rules.map(rule => {
			allGatesNetwork.activate(
				[
					gateCodes[gate.type],
					rule[0][0],
					rule[0][1],
				]
			);

			allGatesNetwork.propagate(
				learningRate, 
				[rule[1]]
			);
		})
	}
})

// test the network
console.log(allGatesNetwork)
console.log('AND [0,0] -> ', Math.round(allGatesNetwork.activate([gateCodes['AND'], 0, 0])));
console.log('AND [0,1] -> ', Math.round(allGatesNetwork.activate([gateCodes['AND'], 0, 1])));
console.log('AND [1,0] -> ', Math.round(allGatesNetwork.activate([gateCodes['AND'], 1, 0])));
console.log('AND [1,1] -> ', Math.round(allGatesNetwork.activate([gateCodes['AND'], 1, 1])));

console.log('XOR [0,0] -> ', Math.round(allGatesNetwork.activate([gateCodes['XOR'], 0, 0])));
console.log('XOR [0,1] -> ', Math.round(allGatesNetwork.activate([gateCodes['XOR'], 0, 1])));
console.log('XOR [1,0] -> ', Math.round(allGatesNetwork.activate([gateCodes['XOR'], 1, 0])));
console.log('XOR [1,1] -> ', Math.round(allGatesNetwork.activate([gateCodes['XOR'], 1, 1])));

console.log('OR [0,0] -> ', Math.round(allGatesNetwork.activate([gateCodes['OR'], 0, 0])));
console.log('OR [0,1] -> ', Math.round(allGatesNetwork.activate([gateCodes['OR'], 0, 1])));
console.log('OR [1,0] -> ', Math.round(allGatesNetwork.activate([gateCodes['OR'], 1, 0])));
console.log('OR [1,1] -> ', Math.round(allGatesNetwork.activate([gateCodes['OR'], 1, 1])));

console.log('NAND [0,0] -> ', Math.round(allGatesNetwork.activate([gateCodes['NAND'], 0, 0])));
console.log('NAND [0,1] -> ', Math.round(allGatesNetwork.activate([gateCodes['NAND'], 0, 1])));
console.log('NAND [1,0] -> ', Math.round(allGatesNetwork.activate([gateCodes['NAND'], 1, 0])));
console.log('NAND [1,1] -> ', Math.round(allGatesNetwork.activate([gateCodes['NAND'], 1, 1])));

console.log('NOR [0,0] -> ', Math.round(allGatesNetwork.activate([gateCodes['NOR'], 0, 0])));
console.log('NOR [0,1] -> ', Math.round(allGatesNetwork.activate([gateCodes['NOR'], 0, 1])));
console.log('NOR [1,0] -> ', Math.round(allGatesNetwork.activate([gateCodes['NOR'], 1, 0])));
console.log('NOR [1,1] -> ', Math.round(allGatesNetwork.activate([gateCodes['NOR'], 1, 1])));