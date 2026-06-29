const { testExperiment } = require("../test-helpers");
const experiments = require("../twoInputGates");

experiments.forEach((experiment) => testExperiment(experiment));
