import { execSync }                                           from "child_process";
import { createOpenAIEmbeddingAdapter }                       from "./openai.js";
import { createGeminiEmbeddingAdapter, createGeminiGenerationAdapter } from "./gemini.js";

// ── Config helpers ────────────────────────────────────────────────────────────

function resolveKey(providerCfg) {
  if (providerCfg.apiKeyCmd) return execSync(providerCfg.apiKeyCmd, { encoding: "utf8" }).trim();
  if (providerCfg.apiKey)    return providerCfg.apiKey;
  throw new Error(`Provider "${providerCfg.type}" is missing apiKey or apiKeyCmd`);
}

function getProviderAndTask(cfg, taskName) {
  const taskCfg = cfg.tasks?.[taskName];
  if (!taskCfg) throw new Error(`Unknown task: "${taskName}"`);

  const providerName = taskCfg.provider;
  const rawProvider  = cfg.providers?.[providerName];
  if (!rawProvider) throw new Error(`Unknown provider: "${providerName}" (task: "${taskName}")`);

  const providerCfg = { ...rawProvider, apiKey: resolveKey(rawProvider) };
  return { providerCfg, taskCfg };
}

// ── Public factories ──────────────────────────────────────────────────────────

// Returns { embed(text), embedBatch(texts) }
export function createEmbeddingAdapter(cfg, taskName = "embedding") {
  const { providerCfg, taskCfg } = getProviderAndTask(cfg, taskName);
  switch (providerCfg.type) {
    case "openai": return createOpenAIEmbeddingAdapter(providerCfg, taskCfg);
    case "gemini": return createGeminiEmbeddingAdapter(providerCfg, taskCfg);
    default: throw new Error(`Unsupported provider type for embedding: "${providerCfg.type}"`);
  }
}

// Returns { generate(system, user, extraConfig), stream(system, user, extraConfig, onChunk) }
export function createGenerationAdapter(cfg, taskName) {
  const { providerCfg, taskCfg } = getProviderAndTask(cfg, taskName);
  switch (providerCfg.type) {
    case "gemini": return createGeminiGenerationAdapter(providerCfg, taskCfg);
    default: throw new Error(`Unsupported provider type for generation: "${providerCfg.type}"`);
  }
}
