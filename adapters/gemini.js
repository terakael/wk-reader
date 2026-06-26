import { l2normalize } from "./utils.js";

const RETRY_LIMIT    = 3;
const RETRY_DELAY_MS = 2000;

function authHeaders(providerCfg, apiKey) {
  const header = providerCfg.authHeader || "x-goog-api-key";
  const value  = header === "Authorization" ? `Bearer ${apiKey}` : apiKey;
  return { [header]: value, "Content-Type": "application/json" };
}

// ── Embedding ─────────────────────────────────────────────────────────────────

export function createGeminiEmbeddingAdapter(providerCfg, taskCfg) {
  const { baseUrl, apiKey }                                          = providerCfg;
  const { model, dimensions, taskType, queryPrefix = "", documentPrefix = "" } = taskCfg;

  const headers = authHeaders(providerCfg, apiKey);

  function buildRequest(text) {
    const req = {
      model:   `models/${model}`,
      content: { parts: [{ text }] },
    };
    if (dimensions) req.output_dimensionality = dimensions;
    if (taskType)   req.task_type             = taskType;
    return req;
  }

  async function fetchBatch(texts, attempt = 1) {
    const res = await fetch(`${baseUrl}/${model}:batchEmbedContents`, {
      method: "POST",
      headers,
      body: JSON.stringify({ requests: texts.map(buildRequest) }),
    });

    if (!res.ok) {
      const err = await res.text();
      if (attempt < RETRY_LIMIT) {
        await new Promise(r => setTimeout(r, RETRY_DELAY_MS * attempt));
        return fetchBatch(texts, attempt + 1);
      }
      throw new Error(`Gemini embed ${res.status}: ${err}`);
    }

    const json = await res.json();
    // gemini-embedding-2 auto-normalises truncated dimensions; we normalise
    // anyway for consistency with the OpenAI adapter.
    return json.embeddings.map(e => l2normalize(new Float32Array(e.values)));
  }

  return {
    async embed(text) {
      const [vec] = await fetchBatch([queryPrefix + text]);
      return vec;
    },

    async embedBatch(texts) {
      return fetchBatch(texts.map(t => documentPrefix + t));
    },
  };
}

// ── Generation ────────────────────────────────────────────────────────────────

export function createGeminiGenerationAdapter(providerCfg, taskCfg) {
  const { baseUrl, apiKey } = providerCfg;
  const { model }           = taskCfg;

  const headers = authHeaders(providerCfg, apiKey);

  return {
    async generate(system, user, extraConfig = {}) {
      const res = await fetch(`${baseUrl}/${model}:generateContent`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          systemInstruction: { parts: [{ text: system }] },
          contents:          [{ role: "user", parts: [{ text: user }] }],
          generationConfig:  { maxOutputTokens: 500, ...extraConfig },
        }),
      });
      if (!res.ok) throw new Error(`Gemini generate ${model} ${res.status}: ${await res.text()}`);
      const json = await res.json();
      return json.candidates[0].content.parts[0].text.trim();
    },

    async stream(system, user, extraConfig = {}, onChunk) {
      const res = await fetch(`${baseUrl}/${model}:streamGenerateContent?alt=sse`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          systemInstruction: { parts: [{ text: system }] },
          contents:          [{ role: "user", parts: [{ text: user }] }],
          generationConfig:  extraConfig,
        }),
      });
      if (!res.ok) throw new Error(`Gemini stream ${model} ${res.status}: ${await res.text()}`);

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        const lines = buf.split("\n");
        buf = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const payload = JSON.parse(line.slice(5).trim());
          const text    = payload.candidates?.[0]?.content?.parts?.[0]?.text;
          if (text) onChunk(text);
        }
      }
    },
  };
}
