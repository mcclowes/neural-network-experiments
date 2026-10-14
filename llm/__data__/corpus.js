// Toy corpus for the char-level LM. Small and highly structured so a tiny MLP
// can memorize it, which (with seeded, deterministic training) lets the spec
// assert exact generated text — the LM analog of the gate truth tables.
const corpus = "the quick brown fox jumps over the lazy dog. ";

module.exports = { corpus };
