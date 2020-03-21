const data = require('../__data__/data')
const ticTacToeNetwork = require('../index')

describe('results', () => {
  const results = data.training.map((example, i) => {
    const result = ticTacToeNetwork.activate(example.input)
      .map(x=> Math.round(x))

    test('test', () => {
      expect(result).toEqual(example.output);
    })
  })
})