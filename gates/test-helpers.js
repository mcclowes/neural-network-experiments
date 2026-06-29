// Shared jest harness for gate experiments. Training is deterministic (seeded
// init + full-batch), so each case is asserted exactly — no retry loop.
const testExperiment = (experiment, { timeout = 30000 } = {}) => {
  describe(experiment.name, () => {
    let model;
    beforeAll(async () => {
      ({ model } = await experiment.train());
    }, timeout);

    afterAll(() => model && model.dispose());

    experiment.cases.forEach((c) => {
      test(c.label || `[${c.input}] -> ${c.output}`, () => {
        expect(experiment.predict(model, c.input)).toEqual(c.output);
      });
    });
  });
};

module.exports = { testExperiment };
