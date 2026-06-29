// Compat shim: @tensorflow/tfjs-node 4.22's native backend calls
// util.isNullOrUndefined, which Node removed in v23. Restore it before the
// backend loads so tfjs-node works on Node 23+. Remove once tfjs-node ships
// a fix. See: https://github.com/tensorflow/tfjs/issues/8266
const util = require("util");
if (typeof util.isNullOrUndefined !== "function") {
  util.isNullOrUndefined = (v) => v === null || v === undefined;
}

const tf = require("@tensorflow/tfjs-node");

// Shared toolkit for the gate experiments. Every gate is a tiny binary
// classifier, so they differ only in their data and layer sizes — that config
// lives in each experiment file; the mechanics live here.

const DEFAULT_SEED = 42;

// Seeded weight init + full-batch training (our datasets are <=20 rows, well
// under the default batch size) makes training deterministic, so tests can
// assert exact results without a retry loop.
const buildModel = (inputShape, layers, { learningRate = 0.1, seed = DEFAULT_SEED } = {}) => {
  const model = tf.sequential();
  layers.forEach((layer, i) => {
    model.add(
      tf.layers.dense({
        units: layer.units,
        activation: layer.activation,
        ...(i === 0 ? { inputShape } : {}),
        kernelInitializer: tf.initializers.glorotUniform({ seed: seed + i }),
      })
    );
  });
  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
};

const trainBinary = async ({ inputs, outputs, inputShape, layers, learningRate, epochs = 300, seed }) => {
  const xs = tf.tensor2d(inputs);
  const ys = tf.tensor2d(outputs);
  const model = buildModel(inputShape, layers, { learningRate, seed });
  const history = await model.fit(xs, ys, { epochs, shuffle: false, verbose: 0 });
  xs.dispose();
  ys.dispose();
  return { model, history };
};

const predict = (model, input) =>
  tf.tidy(() => Math.round(model.predict(tf.tensor2d([input])).dataSync()[0]));

// Builds a gate experiment from pure config. `cases` is [{ input, output, label? }].
const makeExperiment = ({ name, cases, inputShape, layers, learningRate, epochs }) => {
  const train = (opts = {}) =>
    trainBinary({
      inputs: cases.map((c) => c.input),
      outputs: cases.map((c) => [c.output]),
      inputShape,
      layers,
      learningRate,
      epochs,
      ...opts,
    });

  const run = async () => {
    const { model } = await train();
    console.log(name);
    for (const c of cases) {
      const got = predict(model, c.input);
      const ok = got === c.output ? "✓" : "✗";
      console.log(`${ok} ${c.label || `[${c.input}]`} (got ${got})`);
    }
    model.dispose();
  };

  return { name, cases, train, predict, run };
};

module.exports = { tf, buildModel, trainBinary, predict, makeExperiment, DEFAULT_SEED };
