import express from "express";
import { readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import kuromoji from "kuromoji";

const __dirname = dirname(fileURLToPath(import.meta.url));

// ── Config ────────────────────────────────────────────────────────────────────

const PORT         = process.env.PORT || 3000;
const AI_API_KEY  = process.env.AI_API_KEY;
const AI_BASE_URL  = process.env.AI_BASE_URL || "https://api.ai.public.rakuten-it.com";
const EMBED_URL    = `${AI_BASE_URL}/openai/v1/embeddings`;
const GEMINI_BASE  = `${AI_BASE_URL}/google-vertexai/v1/publishers/google/models`;
const EMBED_DIM    = 1536;

// ── Startup data ──────────────────────────────────────────────────────────────

const vocabByLevel = JSON.parse(readFileSync(join(__dirname, "vocabulary.json"), "utf8"));
const kanjiByLevel = JSON.parse(readFileSync(join(__dirname, "kanji.json"), "utf8"));

const embIndex = JSON.parse(readFileSync(join(__dirname, "vocab_embeddings_index.json"), "utf8"));
const embBuf   = readFileSync(join(__dirname, "vocab_embeddings.bin"));
const embVecs  = new Float32Array(embBuf.buffer, embBuf.byteOffset, embBuf.byteLength / 4);

const tokenizer = await new Promise((resolve, reject) =>
  kuromoji
    .builder({ dicPath: join(__dirname, "node_modules/kuromoji/dict") })
    .build((err, t) => (err ? reject(err) : resolve(t)))
);
console.log("Ready.");

// ── Gemini ────────────────────────────────────────────────────────────────────

async function geminiOnce(model, system, user, extraConfig = {}) {
  const res = await fetch(`${GEMINI_BASE}/${model}:generateContent`, {
    method: "POST",
    headers: { Authorization: AI_API_KEY, "Content-Type": "application/json" },
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: system }] },
      contents: [{ role: "user", parts: [{ text: user }] }],
      generationConfig: { maxOutputTokens: 500, ...extraConfig },
    }),
  });
  if (!res.ok) throw new Error(`Gemini ${model} ${res.status}: ${await res.text()}`);
  const json = await res.json();
  return json.candidates[0].content.parts[0].text.trim();
}

// Calls onChunk(text) for each streamed chunk of text.
async function geminiStream(model, system, user, extraConfig = {}, onChunk) {
  const res = await fetch(`${GEMINI_BASE}/${model}:streamGenerateContent?alt=sse`, {
    method: "POST",
    headers: { Authorization: AI_API_KEY, "Content-Type": "application/json" },
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: system }] },
      contents: [{ role: "user", parts: [{ text: user }] }],
      generationConfig: extraConfig,
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
    buf = lines.pop(); // hold incomplete line

    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const payload = JSON.parse(line.slice(5).trim());
      const text = payload.candidates?.[0]?.content?.parts?.[0]?.text;
      if (text) onChunk(text);
    }
  }
}

// ── Embeddings ────────────────────────────────────────────────────────────────

function l2normalize(arr) {
  let norm = 0;
  for (const v of arr) norm += v * v;
  norm = Math.sqrt(norm);
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = arr[i] / norm;
  return out;
}

async function embed(text) {
  const res = await fetch(EMBED_URL, {
    method: "POST",
    headers: { Authorization: `Bearer ${AI_API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify({ input: text, model: "text-embedding-3-small" }),
  });
  if (!res.ok) throw new Error(`Embed ${res.status}: ${await res.text()}`);
  const json = await res.json();
  return l2normalize(new Float32Array(json.data[0].embedding));
}

// Returns top 20 by cosine similarity + 5 sampled from ranks 21–80,
// all filtered to levels ≤ maxLevel.
function retrieveVocab(queryVec, maxLevel) {
  const rowCount = embIndex.maxRowByLevel[maxLevel] ?? embIndex.meta.count;
  const scores = new Array(rowCount);

  for (let r = 0; r < rowCount; r++) {
    const base = r * EMBED_DIM;
    let dot = 0;
    for (let i = 0; i < EMBED_DIM; i++) dot += queryVec[i] * embVecs[base + i];
    scores[r] = { r, dot };
  }
  scores.sort((a, b) => b.dot - a.dot);

  const top20 = scores.slice(0, 20).map(s => embIndex.items[s.r]);

  // Sample 5 from ranks 21–80
  const pool = scores.slice(20, 80);
  const wildcards = [];
  while (wildcards.length < 5 && pool.length > 0) {
    const i = Math.floor(Math.random() * pool.length);
    wildcards.push(embIndex.items[pool.splice(i, 1)[0].r]);
  }

  return [...top20, ...wildcards];
}

// ── Tokeniser ─────────────────────────────────────────────────────────────────

function tokenize(text) {
  return tokenizer.tokenize(text).map(t => {
    let dict = t.basic_form || t.surface_form;
    // Normalise な-adj surface forms: 大きな → 大きい
    if (dict.endsWith("な") && /[ぁ-ん]/.test(dict.slice(-2, -1)))
      dict = dict.slice(0, -1) + "い";
    return { surface: t.surface_form, dict, reading: t.reading || "" };
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function itemsUpToLevel(byLevel, maxLevel) {
  const out = [];
  for (let i = 1; i <= maxLevel; i++)
    for (const item of byLevel[String(i)] || [])
      out.push({ ...item, level: i });
  return out;
}

// ── Express ───────────────────────────────────────────────────────────────────

const app = express();
app.use(express.json());
app.use(express.static(join(__dirname, "public")));

// Vocab + kanji data for the front-end annotation layer.
app.get("/api/data/:maxLevel", (req, res) => {
  const maxLevel = Math.min(60, Math.max(1, parseInt(req.params.maxLevel) || 1));
  res.json({
    vocab:    itemsUpToLevel(vocabByLevel, maxLevel),
    kanji:    itemsUpToLevel(kanjiByLevel, maxLevel),
    allVocab: itemsUpToLevel(vocabByLevel, 60),
    allKanji: itemsUpToLevel(kanjiByLevel, 60),
  });
});

// Placeholder prompt generator — picks random vocab words and asks Gemini for a one-sentence story idea.
app.get("/api/placeholder", async (req, res) => {
  const maxLevel = Math.min(60, Math.max(1, parseInt(req.query.level) || 5));

  // Pick 4 random vocab items within the level
  const maxRow = embIndex.maxRowByLevel[maxLevel] ?? embIndex.meta.count;
  const pool   = embIndex.items.slice(0, maxRow);
  const seeds  = [];
  const used   = new Set();
  while (seeds.length < 4 && seeds.length < pool.length) {
    const idx = Math.floor(Math.random() * pool.length);
    if (!used.has(idx)) { used.add(idx); seeds.push(pool[idx]); }
  }
  const wordList = seeds.map(w => `${w.character} (${w.primary_meaning})`).join(", ");

  try {
    const prompt = await geminiOnce(
      "gemini-2.5-flash",
      "You generate one-sentence story prompt ideas for Japanese reading practice. " +
      "Given a few Japanese vocabulary words, write a single natural English sentence describing a story scenario that could feature them. " +
      "Be specific and vivid — name a character type, a setting, or a situation. " +
      "Output only the sentence, no preamble, no quotes.",
      `Vocabulary words: ${wordList}`,
      { thinkingConfig: { thinkingBudget: 0 } }
    );
    res.json({ prompt: prompt.trim() });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/generate", async (req, res) => {
  const { level = 5, prompt: userPrompt = "Write a short story.", context = "" } = req.body;
  const maxLevel     = Math.min(60, Math.max(1, parseInt(level) || 1));
  const vocabLevel   = Math.max(maxLevel, 5); // floor vocab pool at level 5 for coherent stories

  res.setHeader("Content-Type",  "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection",    "keep-alive");
  res.flushHeaders();

  const send = (type, data) => res.write(`data: ${JSON.stringify({ type, data })}\n\n`);

  try {
    send("status", "Finding vocabulary...");

    // Embed the raw prompt and enhance it concurrently — neither depends on the other.
    // Enhancement doesn't get rough-words context this way, but with thinking off
    // it stays grounded enough and we save ~400ms of sequential latency.
    const [, expanded] = await Promise.all([
      embed(userPrompt),
      geminiOnce(
        "gemini-2.5-flash",
        "Expand the user's story prompt into 2–3 sentences of vivid scene description. " +
        "Focus on concrete nouns, actions, emotions, and setting. " +
        "Reply with only the expanded description, no preamble.",
        userPrompt,
        { thinkingConfig: { thinkingBudget: 0 } }
      ).catch(() => userPrompt),
    ]);

    // Re-embed the grounded description → final vocab retrieval
    const queryVec      = await embed(expanded);
    const selectedVocab = retrieveVocab(queryVec, vocabLevel);
    send("words", selectedVocab.map(w => w.character));

    // 3. Stream the story from Gemini 3.5 Flash (no thinking)
    send("status", "Generating...");

    const wordList = selectedVocab.map(w => w.character).join("、");
    const storyContext = context
      ? `The story so far:\n${context}\n\nContinue naturally from where it left off.`
      : `Story theme: ${userPrompt}`;

    const systemPrompt = [
      `You are writing Japanese reading practice for a WaniKani learner.`,
      ``,
      `The learner is at WaniKani level ${maxLevel}. However, the provided vocabulary list below is the complete source of truth for which content words the learner knows.`,
      ``,
      `Your task:`,
      `- Write one short, natural Japanese story of 10–15 sentences.`,
      `- Make it easy to read and easy to understand.`,
      `- Prioritise comprehensibility over richness or style.`,
      ``,
      `Hard vocabulary rule:`,
      `- Every content word in the story must come from the provided vocabulary list.`,
      `- "Content words" includes nouns, verbs, adjectives, adverbs, and set phrases.`,
      `- Do not use synonyms, near-synonyms, paraphrases, or newly invented compounds unless the exact word is in the provided list.`,
      `- Do not convert unknown words into kana-only form to get around this rule.`,
      `- Do not introduce proper nouns, names, or place names unless they are in the provided list.`,
      ``,
      `Allowed exceptions:`,
      `- You may use basic grammar and function words needed to make correct Japanese, such as particles, copula, auxiliary verbs, inflectional endings, and very common function words.`,
      `- Keep these extra words to the absolute minimum.`,
      `- If a sentence would require an out-of-list content word, simplify the sentence or the plot instead.`,
      ``,
      `Style rules:`,
      `- Prefer short, literal sentences.`,
      `- Use an everyday situation with 1–2 characters at most.`,
      `- Avoid poetic phrasing, detailed description, idioms, and complicated transitions.`,
      `- Use as many provided words as fit naturally, but never force awkward usage.`,
      ``,
      `Before answering:`,
      `- Silently check every content word in your draft.`,
      `- If any content word is not in the provided vocabulary list, rewrite that part.`,
      `- Repeat until the story contains no out-of-list content words.`,
      ``,
      `Output only the Japanese story.`,
    ].join("\n");


    let raw = "";
    await geminiStream(
      "gemini-3.5-flash",
      systemPrompt,
      `Known vocabulary list:\n${wordList}\n\n${storyContext}`,
      { thinkingConfig: { thinkingBudget: 0 } },
      chunk => { raw += chunk; send("delta", chunk); }
    );

    // 4. Tokenise for front-end annotation
    send("done", { tokens: tokenize(raw) });

  } catch (err) {
    console.error(err);
    send("error", err.message || "Generation failed");
  } finally {
    res.end();
  }
});

app.listen(PORT, () => console.log(`http://localhost:${PORT}`));
