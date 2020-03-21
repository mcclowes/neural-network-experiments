const and = {
  type: 'AND',
  rules: [
    {
      input: [0,0],
      output: 0,
    },
    {
      input: [0,1],
      output: 0,
    },
    {
      input: [1,0], 
      output: 0,
    },
    {
      input: [1,1], 
      output: 1,
    }
  ],
}

const nand = {
  type: 'NAND',
  rules: [
    {
      input: [0,0],
      output: 1,
    },
    {
      input: [0,1],
      output: 1,
    },
    {
      input: [1,0], 
      output: 1,
    },
    {
      input: [1,1], 
      output: 0,
    }
  ],
}

const xor = {
  type: 'XOR',
  rules: [
    {
      input: [0,0],
      output: 0,
    },
    {
      input: [0,1],
      output: 1,
    },
    {
      input: [1,0], 
      output: 1,
    },
    {
      input: [1,1], 
      output: 0,
    }
  ],
}

const or = {
  type: 'OR',
  rules: [
    {
      input: [0,0],
      output: 0,
    },
    {
      input: [0,1],
      output: 1,
    },
    {
      input: [1,0], 
      output: 1,
    },
    {
      input: [1,1], 
      output: 1,
    }
  ],
}

const nor = {
  type: 'NOR',
  rules: [
    {
      input: [0,0],
      output: 1,
    },
    {
      input: [0,1],
      output: 0,
    },
    {
      input: [1,0], 
      output: 0,
    },
    {
      input: [1,1], 
      output: 0,
    }
  ],
}

const not = {
 type: 'NOT',
 rules: [
    {
      input: [0],
      output: 1,
    },
    {
      input: [1],
      output: 0,
    },
 ],
}


const data = {
  gates: {
    and,
    nand,
    xor,
    nor,
    or,
    not,
  },
  training: [
    //and
    {
      input: [1, 0, 0, 0, 0, and.rules[0].input[0], and.rules[0].input[1]],
      output: [and.rules[0].output],
      description: "AND - [0, 0] -> 0",
    },
    {
      input: [1, 0, 0, 0, 0, and.rules[1].input[0], and.rules[1].input[1]],
      output: [and.rules[1].output],
      description: "AND - [0, 1] -> 0",
    },
    {
      input: [1, 0, 0, 0, 0, and.rules[2].input[0], and.rules[2].input[1]],
      output: [and.rules[2].output],
      description: "AND - [1, 0] -> 0",
    },
    {
      input: [1, 0, 0, 0, 0, and.rules[3].input[0], and.rules[3].input[1]],
      output: [and.rules[3].output],
      description: "AND - [1, 1] -> 1",
    },
    //nand
    {
      input: [0, 1, 0, 0, 0, nand.rules[0].input[0], nand.rules[0].input[1]],
      output: [nand.rules[0].output],
      description: "NAND - [0, 0] -> 1",
    },
    {
      input: [0, 1, 0, 0, 0, nand.rules[1].input[0], nand.rules[1].input[1]],
      output: [nand.rules[1].output],
      description: "NAND - [0, 1] -> 1",
    },
    {
      input: [0, 1, 0, 0, 0, nand.rules[2].input[0], nand.rules[2].input[1]],
      output: [nand.rules[2].output],
      description: "NAND - [1, 0] -> 1",
    },
    {
      input: [0, 1, 0, 0, 0, nand.rules[3].input[0], nand.rules[3].input[1]],
      output: [nand.rules[3].output],
      description: "NAND - [1, 1] -> 0",
    },
    //xor
    {
      input: [0, 0, 1, 0, 0, xor.rules[0].input[0], xor.rules[0].input[1]],
      output: [xor.rules[0].output],
      description: "XOR - [0, 0] -> 0",
    },
    {
      input: [0, 0, 1, 0, 0, xor.rules[1].input[0], xor.rules[1].input[1]],
      output: [xor.rules[1].output],
      description: "XOR - [0, 1] -> 1",
    },
    {
      input: [0, 0, 1, 0, 0, xor.rules[2].input[0], xor.rules[2].input[1]],
      output: [xor.rules[2].output],
      description: "XOR - [1, 0] -> 1",
    },
    {
      input: [0, 0, 1, 0, 0, xor.rules[3].input[0], xor.rules[3].input[1]],
      output: [xor.rules[3].output],
      description: "XOR - [1, 1] -> 0",
    },
    //nor
    {
      input: [0, 0, 0, 1, 0, nor.rules[0].input[0], nor.rules[0].input[1]],
      output: [nor.rules[0].output],
      description: "NOR - [0, 0] -> 1",
    },
    {
      input: [0, 0, 0, 1, 0, nor.rules[1].input[0], nor.rules[1].input[1]],
      output: [nor.rules[1].output],
      description: "NOR - [0, 1] -> 0",
    },
    {
      input: [0, 0, 0, 1, 0, nor.rules[2].input[0], nor.rules[2].input[1]],
      output: [nor.rules[2].output],
      description: "NOR - [1, 0] -> 0",
    },
    {
      input: [0, 0, 0, 1, 0, nor.rules[3].input[0], nor.rules[3].input[1]],
      output: [nor.rules[3].output],
      description: "NOR - [1, 1] -> 0",
    },
    //or
    {
      input: [0, 0, 0, 0, 1, or.rules[0].input[0], or.rules[0].input[1]],
      output: [or.rules[0].output],
      description: "OR - [0, 0] -> 0",
    },
    {
      input: [0, 0, 0, 0, 1, or.rules[1].input[0], or.rules[1].input[1]],
      output: [or.rules[1].output],
      description: "OR - [0, 1] -> 1",
    },
    {
      input: [0, 0, 0, 0, 1, or.rules[2].input[0], or.rules[2].input[1]],
      output: [or.rules[2].output],
      description: "OR - [1, 0] -> 1",
    },
    {
      input: [0, 0, 0, 0, 1, or.rules[3].input[0], or.rules[3].input[1]],
      output: [or.rules[3].output],
      description: "OR - [1, 1] -> 1",
    },
  ],
}

module.exports = data