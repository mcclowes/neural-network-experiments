const { makeCharLM } = require("./lm");
const { corpus } = require("./__data__/corpus");

// A "simple LLM": a char-level MLP that predicts the next character from the
// previous `contextLength` characters, trained from scratch on a toy corpus.
// Generate text by feeding predictions back in (see lm.js `generate`).
const charLM = makeCharLM({
  name: "char-LM",
  text: corpus,
  contextLength: 8,
  hidden: 64,
  learningRate: 0.05,
  epochs: 600,
  // Next-char cases (the LM analog of a gate truth table). Each context is a
  // unique 8-gram from the corpus, so its next char is unambiguous.
  cases: [
    { context: "the quic", next: "k" },
    { context: "brown fo", next: "x" },
    { context: "umps ove", next: "r" },
    { context: "he lazy ", next: "d" },
  ],
});

if (require.main === module) charLM.run();

module.exports = charLM;
