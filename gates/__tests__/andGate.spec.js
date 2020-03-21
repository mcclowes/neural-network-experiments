const data = require('../__data__/allGatesData')
const andGateNetwork = require('../andGate')

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

describe('results', () => {
  const results = trainingSet.map((example, i) => {
    const result = andGateNetwork.activate(example.input)
    test(example.description, () => {
      expect(Math.round(result)).toEqual(example.output[0]);
    })
  })
})