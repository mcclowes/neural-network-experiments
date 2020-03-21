const data = require('../__data__/allGatesData')
const allGatesNetwork = require('../allGates')

describe('results', () => {
  const results = data.training.map((example, i) => {
    const result = allGatesNetwork.activate(example.input)
    test(example.description, () => {
      expect(Math.round(result)).toEqual(example.output[0]);
    })
  })
})