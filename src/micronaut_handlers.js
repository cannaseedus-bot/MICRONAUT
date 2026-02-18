/**
 * MICRONAUT HANDLERS
 * Lightweight AI inference using n-gram models
 */

import fs from 'fs-extra';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Brain models cache
let brainModels = {
  trigrams: null,
  bigrams: null,
  intents: null
};

// Load brain models
const loadBrains = async () => {
  const brainsDir = path.resolve(__dirname, '../../micronaut/brains');

  try {
    if (!brainModels.trigrams) {
      const trigramPath = path.join(brainsDir, 'trigrams.json');
      if (await fs.pathExists(trigramPath)) {
        brainModels.trigrams = await fs.readJson(trigramPath);
      }
    }

    if (!brainModels.bigrams) {
      const bigramPath = path.join(brainsDir, 'bigrams.json');
      if (await fs.pathExists(bigramPath)) {
        brainModels.bigrams = await fs.readJson(bigramPath);
      }
    }

    if (!brainModels.intents) {
      const intentPath = path.join(brainsDir, 'meta-intent-map.json');
      if (await fs.pathExists(intentPath)) {
        brainModels.intents = await fs.readJson(intentPath);
      }
    }
  } catch (err) {
    console.warn('Brain models not yet trained:', err.message);
  }

  return brainModels;
};

export default {
  /**
   * Infer - Generate text completion
   */
  micronaut_infer: async ({ input }) => {
    const { prompt, max_tokens = 50, temperature = 0.7 } = input;

    await loadBrains();

    if (!brainModels.trigrams && !brainModels.bigrams) {
      return {
        error: 'Brain models not loaded',
        note: 'Run training scripts in /micronaut/training/',
        completion: ''
      };
    }

    // Generate using n-gram model
    const completion = generateCompletion(prompt, max_tokens, temperature);

    return {
      success: true,
      prompt,
      completion,
      model: 'micronaut-ngram',
      tokens: completion.split(/\s+/).length
    };
  },

  /**
   * Intent - Classify intent
   */
  micronaut_intent: async ({ input }) => {
    const { text } = input;

    await loadBrains();

    if (!brainModels.intents) {
      return {
        error: 'Intent model not loaded',
        intent: 'unknown',
        confidence: 0
      };
    }

    const intent = classifyIntent(text, brainModels.intents);

    return {
      success: true,
      text,
      intent: intent.name,
      target: intent.target,
      confidence: intent.confidence,
      alternatives: intent.alternatives
    };
  },

  /**
   * Route - Classify text and return micronaut routing decision (does not execute)
   */
  micronaut_route: async ({ input }) => {
    const { text } = input;

    await loadBrains();

    if (!brainModels.intents) {
      return {
        error: 'Intent model not loaded',
        target: 'XM-1',
        confidence: 0
      };
    }

    const route = micronautRoute(text, brainModels.intents);

    return {
      success: true,
      text,
      ...route,
    };
  },

  /**
   * Complete - Code completion
   */
  micronaut_complete: async ({ input }) => {
    const { code, language = 'javascript', max_suggestions = 3 } = input;

    await loadBrains();

    // Generate code suggestions
    const suggestions = generateCodeSuggestions(code, language, max_suggestions);

    return {
      success: true,
      code,
      language,
      suggestions
    };
  },

  /**
   * Chat - Conversational response
   */
  micronaut_chat: async ({ input }) => {
    const { message, context = [] } = input;

    await loadBrains();

    // Generate chat response
    const response = generateChatResponse(message, context);

    return {
      success: true,
      message,
      response,
      model: 'micronaut-chat'
    };
  },

  /**
   * Train - Trigger model training
   */
  micronaut_train: async ({ input }) => {
    const { model_type = 'trigrams', dataset } = input;

    return {
      error: 'Training not yet implemented in server',
      note: 'Run training scripts directly: node micronaut/training/train-trigrams.js',
      model_type
    };
  },

  /**
   * Status - Get Micronaut status
   */
  micronaut_status: async ({ input }) => {
    await loadBrains();

    return {
      status: 'online',
      models: {
        trigrams: brainModels.trigrams ? 'loaded' : 'not loaded',
        bigrams: brainModels.bigrams ? 'loaded' : 'not loaded',
        intents: brainModels.intents ? 'loaded' : 'not loaded'
      },
      capabilities: [
        'text_completion',
        'intent_classification',
        'code_completion',
        'chat_response'
      ]
    };
  }
};

// Helper functions
function generateCompletion(prompt, maxTokens, temperature) {
  // Mock completion - in production, use actual n-gram model
  const words = prompt.split(/\s+/);
  const lastWord = words[words.length - 1];

  // Simple completion based on last word
  const completions = {
    'the': 'system is working',
    'function': 'returns a value',
    'create': 'a new instance',
    'implement': 'the feature',
    'use': 'the API endpoint'
  };

  return completions[lastWord.toLowerCase()] || 'completion not available';
}

// -----------------------------------------------------------------------
// Real ngram intent classifier using meta-intent-map.json triggers
// -----------------------------------------------------------------------

function extractBigrams(text) {
  const words = text.toLowerCase().split(/\s+/).filter(Boolean);
  const result = [];
  for (let i = 0; i < words.length - 1; i++) {
    result.push(`${words[i]} ${words[i + 1]}`);
  }
  return result;
}

function extractTrigrams(text) {
  const words = text.toLowerCase().split(/\s+/).filter(Boolean);
  const result = [];
  for (let i = 0; i < words.length - 2; i++) {
    result.push(`${words[i]} ${words[i + 1]} ${words[i + 2]}`);
  }
  return result;
}

function scoreIntent(text, intentDef) {
  const BIGRAM_WEIGHT = 1.0;
  const TRIGRAM_WEIGHT = 1.5;
  const bigrams = extractBigrams(text);
  const trigrams = extractTrigrams(text);
  let score = 0;
  for (const bg of (intentDef.trigger_bigrams || [])) {
    if (bigrams.includes(bg)) score += BIGRAM_WEIGHT;
  }
  for (const tg of (intentDef.trigger_trigrams || [])) {
    if (trigrams.includes(tg)) score += TRIGRAM_WEIGHT;
  }
  return score;
}

function classifyIntent(text, intentMap) {
  const MIN_CONFIDENCE = intentMap?.routing?.minimum_confidence ?? 0.3;
  const FALLBACK = intentMap?.routing?.fallback ?? 'XM-1';
  const intents = intentMap?.intents ?? {};

  const scored = [];
  for (const [name, def] of Object.entries(intents)) {
    const score = scoreIntent(text, def);
    if (score > 0) {
      scored.push({ name, score, def });
    }
  }
  scored.sort((a, b) => b.score - a.score || (a.def.priority ?? 99) - (b.def.priority ?? 99));

  if (scored.length > 0 && scored[0].score >= MIN_CONFIDENCE) {
    const winner = scored[0];
    return {
      name: winner.name,
      target: winner.def.target,
      confidence: winner.score,
      alternatives: scored.slice(1, 4).map(s => s.name),
    };
  }

  return {
    name: 'expand',
    target: FALLBACK,
    confidence: 0,
    alternatives: [],
  };
}

/**
 * micronaut_route — classify text and return routing decision.
 * Does NOT execute any tool; only routes.
 */
function micronautRoute(text, intentMap) {
  const result = classifyIntent(text, intentMap);
  const intents = intentMap?.intents ?? {};
  const intentDef = Object.values(intents).find(d => d.target === result.target) ?? {};
  const tools = intentDef.tools ?? [];
  return {
    target: result.target,
    intent: result.name,
    target_fold: intentDef.fold ?? '',
    tool: tools[0] ?? null,
    confidence: result.confidence,
    alternatives: result.alternatives,
  };
}

function generateCodeSuggestions(code, language, maxSuggestions) {
  // Mock code suggestions
  const suggestions = [];

  if (code.includes('function')) {
    suggestions.push({
      text: 'function name() { }',
      type: 'function',
      confidence: 0.8
    });
  }

  if (code.includes('const')) {
    suggestions.push({
      text: 'const variable = value;',
      type: 'variable',
      confidence: 0.75
    });
  }

  if (code.includes('import')) {
    suggestions.push({
      text: 'import { module } from \'package\';',
      type: 'import',
      confidence: 0.7
    });
  }

  return suggestions.slice(0, maxSuggestions);
}

function generateChatResponse(message, context) {
  // Mock chat response
  const responses = {
    'hello': 'Hello! How can I help you today?',
    'hi': 'Hi there!',
    'help': 'I can assist with code completion, intent classification, and text generation.',
    'thanks': 'You\'re welcome!',
    'bye': 'Goodbye!'
  };

  const lowerMessage = message.toLowerCase();
  for (const [keyword, response] of Object.entries(responses)) {
    if (lowerMessage.includes(keyword)) {
      return response;
    }
  }

  return 'I understand. How can I assist you further?';
}
