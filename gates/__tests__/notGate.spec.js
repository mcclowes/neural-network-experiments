const oneInputNetwork = require('../notGate')

describe('not gate', () => {
  test('works', () => {
    expect(Math.round(oneInputNetwork.activate([0]))).toBe(1);
    expect(Math.round(oneInputNetwork.activate([1]))).toBe(0);
  })
})