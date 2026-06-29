const { testExperiment } = require("../test-helpers");
const allGates = require("../allGates");

testExperiment(allGates, { timeout: 60000 });
