import { l2normalize } from "./utils.js";

const RETRY_LIMIT    = 3;
const RETRY_DELAY_MS = 2000;

// ── Embedding ─────────────────────────────────────────────────────────────────

export function createOpenAIEmbeddingAdapter(providerCfg, taskCfg) {
  const { baseUrl, apiKey }                                 = providerCfg;
  const { model, dimensions, queryPrefix = "", documentPrefix = "" } = taskCfg;

  const headers = {
    Authorization:  `Bearer ${apiKey}`,
    "Content-Type": "application/json",
  };

  async function fetchBatch(texts, attempt = 1) {
    const body = { input: texts, model };
    if (dimensions) body.dimensions = dimensions;

    const res = await fetch(`${baseUrl}/embeddings`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const err = await res.text();
      if (attempt < RETRY_LIMIT) {
        await new Promise(r => setTimeout(r, RETRY_DELAY_MS * attempt));
        return fetchBatch(texts, attempt + 1);
      }
      throw new Error(`OpenAI embed ${res.status}: ${err}`);
    }

    const json = await res.json();
    return json.data
      .sort((a, b) => a.index - b.index)
      .map(d => l2normalize(new Float32Array(d.embedding)));
  }

  return {
    // Single text at query time — applies queryPrefix.
    async embed(text) {
      const [vec] = await fetchBatch([queryPrefix + text]);
      return vec;
    },

    // Batch of texts at index-build time — applies documentPrefix.
    async embedBatch(texts) {
      return fetchBatch(texts.map(t => documentPrefix + t));
    },
  };
}
