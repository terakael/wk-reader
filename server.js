import express from "express";
import { readFileSync, existsSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import kuromoji from "kuromoji";
import { createEmbeddingAdapter, createGenerationAdapter } from "./adapters/index.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

// ── Config ────────────────────────────────────────────────────────────────────

const configPath = join(__dirname, "config.json");
if (!existsSync(configPath)) {
  console.error("config.json not found. Copy config.example.json and fill in your values.");
  process.exit(1);
}
const cfg = JSON.parse(readFileSync(configPath, "utf8"));

const PORT = cfg.port || process.env.PORT || 3000;

const embedAdapter   = createEmbeddingAdapter(cfg, "embedding");
const enhanceAdapter = createGenerationAdapter(cfg, "enhance");
const storyAdapter   = createGenerationAdapter(cfg, "story");

// ── Startup data ──────────────────────────────────────────────────────────────

const vocabByLevel = JSON.parse(readFileSync(join(__dirname, "vocabulary.json"), "utf8"));
const kanjiByLevel = JSON.parse(readFileSync(join(__dirname, "kanji.json"), "utf8"));

const embIndex = JSON.parse(readFileSync(join(__dirname, "vocab_embeddings_index.json"), "utf8"));
const embBuf   = readFileSync(join(__dirname, "vocab_embeddings.bin"));

// Dimensions come from the index — it's the source of truth for what's stored.
const EMBED_DIM = embIndex.meta.dimensions;
const embVecs   = new Float32Array(embBuf.buffer, embBuf.byteOffset, embBuf.byteLength / 4);

const tokenizer = await new Promise((resolve, reject) =>
  kuromoji
    .builder({ dicPath: join(__dirname, "node_modules/kuromoji/dict") })
    .build((err, t) => (err ? reject(err) : resolve(t)))
);
console.log("Ready.");

// ── Helpers ───────────────────────────────────────────────────────────────────

// Returns top 20 by cosine similarity + 5 sampled from ranks 21–80,
// all filtered to levels ≤ maxLevel.
function retrieveVocab(queryVec, maxLevel) {
  const rowCount = embIndex.maxRowByLevel[maxLevel] ?? embIndex.meta.count;
  const scores   = new Array(rowCount);

  for (let r = 0; r < rowCount; r++) {
    const base = r * EMBED_DIM;
    let dot = 0;
    for (let i = 0; i < EMBED_DIM; i++) dot += queryVec[i] * embVecs[base + i];
    scores[r] = { r, dot };
  }
  scores.sort((a, b) => b.dot - a.dot);

  const top20 = scores.slice(0, 20).map(s => embIndex.items[s.r]);

  const pool      = scores.slice(20, 80);
  const wildcards = [];
  while (wildcards.length < 5 && pool.length > 0) {
    const i = Math.floor(Math.random() * pool.length);
    wildcards.push(embIndex.items[pool.splice(i, 1)[0].r]);
  }

  return [...top20, ...wildcards];
}

function tokenize(text) {
  return tokenizer.tokenize(text).map(t => {
    let dict = t.basic_form || t.surface_form;
    if (dict.endsWith("な") && /[ぁ-ん]/.test(dict.slice(-2, -1)))
      dict = dict.slice(0, -1) + "い";
    return { surface: t.surface_form, dict, reading: t.reading || "" };
  });
}

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

app.get("/api/data/:maxLevel", (req, res) => {
  const maxLevel = Math.min(60, Math.max(1, parseInt(req.params.maxLevel) || 1));
  res.json({
    vocab:    itemsUpToLevel(vocabByLevel, maxLevel),
    kanji:    itemsUpToLevel(kanjiByLevel, maxLevel),
    allVocab: itemsUpToLevel(vocabByLevel, 60),
    allKanji: itemsUpToLevel(kanjiByLevel, 60),
  });
});

app.get("/api/placeholder", async (req, res) => {
  const maxLevel = Math.min(60, Math.max(1, parseInt(req.query.level) || 5));

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
    const prompt = await enhanceAdapter.generate(
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
  const maxLevel   = Math.min(60, Math.max(1, parseInt(level) || 1));
  const vocabLevel = Math.max(maxLevel, 5);

  res.setHeader("Content-Type",  "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection",    "keep-alive");
  res.flushHeaders();

  const send = (type, data) => res.write(`data: ${JSON.stringify({ type, data })}\n\n`);

  try {
    send("status", "Finding vocabulary...");

    const pool   = embIndex.items.slice(0, embIndex.maxRowByLevel[vocabLevel] ?? embIndex.meta.count);
    const sample = pool
      .slice().sort(() => Math.random() - 0.5).slice(0, 20)
      .map(w => `${w.character}(${w.primary_meaning})`).join(", ");

    const expanded = await enhanceAdapter.generate(
      "You help generate Japanese story prompts for vocabulary learners. " +
      "Given a story theme and a sample of vocabulary available at the learner's level, " +
      "expand the theme into 2–3 sentences of vivid scene description. " +
      "Find a creative angle on the theme that naturally connects to the available vocabulary words. " +
      "Focus on concrete nouns, actions, emotions, and setting. " +
      "Reply with only the expanded description, no preamble.",
      `Theme: ${userPrompt}\n\nAvailable vocabulary sample: ${sample}`,
      { thinkingConfig: { thinkingBudget: 0 } }
    ).catch(() => userPrompt);

    const queryVec      = await embedAdapter.embed(expanded);
    const selectedVocab = retrieveVocab(queryVec, vocabLevel);
    send("words", selectedVocab.map(w => w.character));

    send("status", "Generating...");

    const wordList     = selectedVocab.map(w => w.character).join("、");
    const storyContext = context
      ? `The story so far:\n${context}\n\nContinue from where it left off — resolve the situation, then end at a new moment of tension. Complete every sentence fully; no ellipses.`
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
      `- Exception: character names, place names, and proper nouns from the story theme may be used freely — write them in katakana.`,
      ``,
      `Allowed exceptions:`,
      `- You may use basic grammar and function words needed to make correct Japanese, such as particles, copula, auxiliary verbs, inflectional endings, and very common function words.`,
      `- Keep these extra words to the absolute minimum.`,
      `- If a sentence would require an out-of-list content word, simplify the sentence or the plot instead.`,
      ``,
      `Style rules:`,
      `- Write clear, natural prose; compound and subordinate clauses are fine where they read naturally.`,
      `- Use an everyday situation with 1–2 characters at most.`,
      `- Avoid poetic phrasing, flowery description, and obscure idioms.`,
      `- Use as many provided words as fit naturally, but never force awkward usage.`,
      ``,
      `Before answering:`,
      `- Silently check every content word in your draft.`,
      `- If any content word is not in the provided vocabulary list, rewrite that part.`,
      `- Repeat until the story contains no out-of-list content words.`,
      ``,
      `Output only the Japanese story. End the segment at a moment of genuine narrative tension — complete your final sentence, but leave the situation unresolved in a way that makes the next development feel inevitable. Do not use ellipses or trailing off; the sentence should be whole.`,
    ].join("\n");

    let raw = "";
    await storyAdapter.stream(
      systemPrompt,
      `Known vocabulary list:\n${wordList}\n\n${storyContext}`,
      { thinkingConfig: { thinkingBudget: 0 } },
      chunk => { raw += chunk; send("delta", chunk); }
    );

    send("done", { tokens: tokenize(raw) });

  } catch (err) {
    console.error(err);
    send("error", err.message || "Generation failed");
  } finally {
    res.end();
  }
});

app.listen(PORT, "0.0.0.0", () => console.log(`http://localhost:${PORT}`));
