(function (Scratch) {
  "use strict";
  if (!Scratch.extensions.unsandboxed) {
    throw new Error("TinyGPT requires TurboWarp's unsandboxed custom extension mode.");
  }

  const MODEL = __MODEL_DATA__;
  const D = MODEL.dimensions.embedding;
  const HEADS = MODEL.dimensions.heads;
  const HEAD_WIDTH = MODEL.dimensions.headWidth;
  const FFN = MODEL.dimensions.ffn;
  const LAYERS = MODEL.dimensions.layers;
  const CONTEXT = MODEL.dimensions.context;
  const VOCAB = MODEL.vocab.length;
  let weights = null;

  function loadWeights() {
    if (weights) return;
    const binary = atob(MODEL.weights);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index++) {
      bytes[index] = binary.charCodeAt(index);
    }
    weights = new Int8Array(bytes.buffer);
  }

  function updateVariable(name, value) {
    const id = MODEL.variables[name];
    for (const target of Scratch.vm.runtime.targets) {
      const variable = target.variables && target.variables[id];
      if (variable) {
        variable.value = value;
        Scratch.vm.runtime.requestRedraw();
        return;
      }
    }
  }

  function progressBar(percent) {
    const cells = Math.max(0, Math.min(20, Math.floor(percent / 5)));
    return "\u2588".repeat(cells) + "\u2591".repeat(20 - cells);
  }

  function layerNorm(input, prefix) {
    const gamma = MODEL.tensors[prefix + ".weight"];
    const beta = MODEL.tensors[prefix + ".bias"];
    let mean = 0;
    for (let index = 0; index < input.length; index++) mean += input[index];
    mean /= input.length;
    let variance = 0;
    for (let index = 0; index < input.length; index++) {
      const difference = input[index] - mean;
      variance += difference * difference;
    }
    variance /= input.length;
    const inverseStd = 1 / Math.sqrt(variance + 1e-5);
    const output = new Float32Array(input.length);
    for (let index = 0; index < input.length; index++) {
      const scale = weights[gamma.offset + index] * gamma.scale;
      const shift = weights[beta.offset + index] * beta.scale;
      output[index] = (input[index] - mean) * inverseStd * scale + shift;
    }
    return output;
  }

  function linear(input, weightName, biasName, outputSize) {
    const weight = MODEL.tensors[weightName];
    const inputSize = input.length;
    const scaledInput = new Float32Array(inputSize);
    for (let column = 0; column < inputSize; column++) {
      scaledInput[column] = input[column] * weight.scale;
    }
    const bias = biasName ? MODEL.tensors[biasName] : null;
    const output = new Float32Array(outputSize);
    for (let row = 0; row < outputSize; row++) {
      let sum = bias ? weights[bias.offset + row] * bias.scale : 0;
      const rowOffset = weight.offset + row * inputSize;
      for (let column = 0; column < inputSize; column++) {
        sum += weights[rowOffset + column] * scaledInput[column];
      }
      output[row] = sum;
    }
    return output;
  }

  function gelu(value) {
    const inner = Math.sqrt(2 / Math.PI) * (value + 0.044715 * value * value * value);
    return 0.5 * value * (1 + Math.tanh(inner));
  }

  function sample(logits, temperature) {
    let maximum = -Infinity;
    for (let index = 0; index < logits.length; index++) {
      maximum = Math.max(maximum, logits[index]);
    }
    const probabilities = new Float64Array(logits.length);
    let total = 0;
    for (let index = 0; index < logits.length; index++) {
      const probability = Math.exp((logits[index] - maximum) / temperature);
      probabilities[index] = probability;
      total += probability;
    }
    let threshold = Math.random() * total;
    for (let index = 0; index < probabilities.length; index++) {
      threshold -= probabilities[index];
      if (threshold <= 0) return index;
    }
    return probabilities.length - 1;
  }

  function createForward(cacheKeys, cacheValues, maxTokens) {
    return async function forwardToken(tokenId, position, generationStep, phase) {
      const embedding = MODEL.tensors["token_embedding.weight"];
      const positionEmbedding = MODEL.tensors["position_embedding.weight"];
      const tokenOffset = embedding.offset + tokenId * D;
      const positionOffset = positionEmbedding.offset + position * D;
      let state = new Float32Array(D);
      for (let column = 0; column < D; column++) {
        state[column] = weights[tokenOffset + column] * embedding.scale +
          weights[positionOffset + column] * positionEmbedding.scale;
      }

      for (let layer = 0; layer < LAYERS; layer++) {
        const prefix = `blocks.${layer}.`;
        const normalized = layerNorm(state, prefix + "ln1");
        const qkv = linear(normalized, prefix + "attention.qkv.weight", prefix + "attention.qkv.bias", 3 * D);
        const cacheBase = layer * CONTEXT * D + position * D;
        for (let column = 0; column < D; column++) {
          cacheKeys[cacheBase + column] = qkv[D + column];
          cacheValues[cacheBase + column] = qkv[2 * D + column];
        }

        const attended = new Float32Array(D);
        const scores = new Float64Array(position + 1);
        const layerCacheBase = layer * CONTEXT * D;
        for (let head = 0; head < HEADS; head++) {
          const headOffset = head * HEAD_WIDTH;
          let maximum = -Infinity;
          for (let keyPosition = 0; keyPosition <= position; keyPosition++) {
            let score = 0;
            const keyOffset = layerCacheBase + keyPosition * D + headOffset;
            for (let column = 0; column < HEAD_WIDTH; column++) {
              score += qkv[headOffset + column] * cacheKeys[keyOffset + column];
            }
            score /= Math.sqrt(HEAD_WIDTH);
            scores[keyPosition] = score;
            maximum = Math.max(maximum, score);
          }
          let denominator = 0;
          for (let keyPosition = 0; keyPosition <= position; keyPosition++) {
            scores[keyPosition] = Math.exp(scores[keyPosition] - maximum);
            denominator += scores[keyPosition];
          }
          for (let column = 0; column < HEAD_WIDTH; column++) {
            let sum = 0;
            for (let keyPosition = 0; keyPosition <= position; keyPosition++) {
              const valueOffset = layerCacheBase + keyPosition * D + headOffset;
              sum += (scores[keyPosition] / denominator) * cacheValues[valueOffset + column];
            }
            attended[headOffset + column] = sum;
          }
        }

        const projected = linear(attended, prefix + "attention.proj.weight", prefix + "attention.proj.bias", D);
        for (let column = 0; column < D; column++) state[column] += projected[column];
        const normalizedFfn = layerNorm(state, prefix + "ln2");
        const hidden = linear(normalizedFfn, prefix + "ffn.net.0.weight", prefix + "ffn.net.0.bias", FFN);
        for (let index = 0; index < FFN; index++) hidden[index] = gelu(hidden[index]);
        const output = linear(hidden, prefix + "ffn.net.2.weight", prefix + "ffn.net.2.bias", D);
        for (let column = 0; column < D; column++) state[column] += output[column];

        if (phase === "generation") {
          const percent = ((generationStep + (layer + 1) / LAYERS) / maxTokens) * 100;
          updateVariable("generationProgress", percent);
          updateVariable("progressBar", progressBar(percent));
          updateVariable("progressText", `Token ${generationStep + 1}/${maxTokens}, layer ${layer + 1}/${LAYERS}`);
          await new Promise(resolve => setTimeout(resolve, 0));
        } else {
          updateVariable("progressText", `Prefill token ${position + 1}, layer ${layer + 1}/${LAYERS}`);
          await new Promise(resolve => setTimeout(resolve, 0));
        }
      }
      return linear(layerNorm(state, "ln_final"), "token_embedding.weight", null, VOCAB);
    };
  }

  class TinyGPTFast {
    getInfo() {
      return {
        id: "tinygpt",
        name: "TinyGPT Fast",
        blocks: [{
          opcode: "generate",
          blockType: Scratch.BlockType.REPORTER,
          text: "generate story from [PROMPT] max [MAX_TOKENS] tokens at temperature [TEMPERATURE]",
          arguments: {
            PROMPT: { type: Scratch.ArgumentType.STRING, defaultValue: "once upon a time" },
            MAX_TOKENS: { type: Scratch.ArgumentType.NUMBER, defaultValue: MODEL.dimensions.defaultTokens },
            TEMPERATURE: { type: Scratch.ArgumentType.NUMBER, defaultValue: MODEL.dimensions.temperature }
          }
        }]
      };
    }

    async generate(args) {
      try {
        loadWeights();
        const maximumTokens = Math.max(1, Math.min(50, Math.floor(Number(args.MAX_TOKENS) || MODEL.dimensions.defaultTokens)));
        const temperature = Math.max(0.01, Number(args.TEMPERATURE) || MODEL.dimensions.temperature);
        const wordToId = new Map(MODEL.vocab.map((word, index) => [word, index]));
        const input = String(args.PROMPT).toLowerCase().match(/[\p{L}\p{N}_]+/gu) || [];
        let tokens = input.map(word => wordToId.has(word) ? wordToId.get(word) : 1);
        tokens = tokens.slice(-Math.max(1, CONTEXT - maximumTokens));
        if (!tokens.length) tokens = [0];
        let story = String(args.PROMPT);
        updateVariable("storyText", story);
        updateVariable("generationProgress", 0);
        updateVariable("progressBar", progressBar(0));
        updateVariable("progressText", "Prefilling prompt");

        const cacheKeys = new Float32Array(LAYERS * CONTEXT * D);
        const cacheValues = new Float32Array(LAYERS * CONTEXT * D);
        const forwardToken = createForward(cacheKeys, cacheValues, maximumTokens);
        let position = 0;
        let logits;
        for (const token of tokens) {
          logits = await forwardToken(token, position, 0, "prefill");
          position++;
        }

        for (let step = 0; step < maximumTokens; step++) {
          const token = sample(logits, temperature);
          const word = MODEL.vocab[token] || "<UNK>";
          story += (story ? " " : "") + word;
          updateVariable("storyText", story);
          updateVariable("generationProgress", ((step + 1) / maximumTokens) * 100);
          updateVariable("progressBar", progressBar(((step + 1) / maximumTokens) * 100));
          updateVariable("progressText", `Generated ${step + 1}/${maximumTokens} tokens`);
          await new Promise(resolve => setTimeout(resolve, 0));
          if (step + 1 < maximumTokens) {
            logits = await forwardToken(token, position, step + 1, "generation");
            position++;
          }
        }
        updateVariable("progressText", "Story complete");
        return story;
      } catch (error) {
        updateVariable("progressText", `Model error: ${error.message}`);
        return `TinyGPT error: ${error.message}`;
      }
    }
  }

  Scratch.extensions.register(new TinyGPTFast());
})(Scratch);