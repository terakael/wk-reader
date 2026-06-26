#!/usr/bin/env node
/**
 * One-time build script: embeds all WaniKani vocab items and writes:
 *   vocab_embeddings.bin          — Float32Array, L2-normalised, sorted by level (row-major)
 *   vocab_embeddings_index.json   — metadata, per-item index, maxRowByLevel lookup
 *
 * Usage:
 *   node scripts/embed-vocab.js
 *
 * Reads embedding.url, embedding.model, and embedding.apiKey / embedding.apiKeyCmd from config.json.
 */

import { readFileSync, writeFileSync, existsSync } from "fs";
import { execSync } from "child_process";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");

const configPath = join(ROOT, "config.json");
if (!existsSync(configPath)) {
  console.error("config.json not found. Copy config.example.json and fill in your values.");
  process.exit(1);
}
const cfg = JSON.parse(readFileSync(configPath, "utf8"));

function resolve(section, field) {
  const cmdKey = `${field}Cmd`;
  if (section[cmdKey]) return execSync(section[cmdKey], { encoding: "utf8" }).trim();
  if (section[field])  return section[field];
  throw new Error(`config.json: missing "${field}" or "${cmdKey}"`);
}

const API_URL = cfg.embedding.url;
const MODEL   = cfg.embedding.model || "text-embedding-3-small";
const apiKey  = resolve(cfg.embedding, "apiKey");
const DIMENSIONS = 1536;
const BATCH_SIZE = 100;
const RETRY_LIMIT = 3;
const RETRY_DELAY_MS = 2000;

// ── Load + sort vocab ─────────────────────────────────────────────────────────

const vocabByLevel = JSON.parse(readFileSync(join(ROOT, "vocabulary.json"), "utf8"));

const items = [];
for (let level = 1; level <= 60; level++) {
  for (const item of vocabByLevel[String(level)] || []) {
    items.push({
      character: item.character,
      reading: item.reading || "",
      primary_meaning: item.primary_meaning || item.meaning || "",
      alternative_meanings: item.alternative_meanings || [],
      level,
      url: item.url || "",
    });
  }
}
// Sort ascending by level so rows 0..maxRowByLevel[N] covers all levels ≤ N
items.sort((a, b) => a.level - b.level);

console.log(`Loaded ${items.length} vocab items across 60 levels.`);

// ── Build embedding text ──────────────────────────────────────────────────────

function embeddingText(item) {
  const alts = item.alternative_meanings.slice(0, 4).join(", ");
  const meanings = alts ? `${item.primary_meaning}, ${alts}` : item.primary_meaning;
  return `Word: ${item.character}\nReading: ${item.reading}\nMeanings: ${meanings}`;
}

// ── Embed helpers ─────────────────────────────────────────────────────────────


async function embedBatch(texts, attempt = 1) {
  const res = await fetch(API_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ input: texts, model: MODEL }),
  });

  if (!res.ok) {
    const body = await res.text();
    if (attempt < RETRY_LIMIT) {
      console.warn(`  Batch failed (${res.status}), retrying in ${RETRY_DELAY_MS}ms… [${body.slice(0,120)}]`);
      await new Promise((r) => setTimeout(r, RETRY_DELAY_MS * attempt));
      return embedBatch(texts, attempt + 1);
    }
    throw new Error(`Embedding API error ${res.status}: ${body}`);
  }

  const json = await res.json();
  // API returns data sorted by index
  return json.data.sort((a, b) => a.index - b.index).map((d) => d.embedding);
}

function l2normalize(vec) {
  let norm = 0;
  for (const v of vec) norm += v * v;
  norm = Math.sqrt(norm);
  return norm > 0 ? vec.map((v) => v / norm) : vec;
}

// ── Main ──────────────────────────────────────────────────────────────────────

const allVectors = new Float32Array(items.length * DIMENSIONS);
let totalTokens = 0;

const batches = Math.ceil(items.length / BATCH_SIZE);
console.log(`Embedding in ${batches} batches of up to ${BATCH_SIZE}…\n`);

for (let b = 0; b < batches; b++) {
  const start = b * BATCH_SIZE;
  const end = Math.min(start + BATCH_SIZE, items.length);
  const slice = items.slice(start, end);
  const texts = slice.map(embeddingText);

  process.stdout.write(`  Batch ${b + 1}/${batches} (items ${start}–${end - 1})… `);

  const res = await fetch(API_URL, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify({ input: texts, model: MODEL }),
  });

  if (!res.ok) {
    const body = await res.text();
    if (b > 0) {
      // save progress before dying
      console.error(`\nFailed at batch ${b + 1}. Partial output not saved.`);
    }
    throw new Error(`API error ${res.status}: ${body.slice(0, 200)}`);
  }

  const json = await res.json();
  totalTokens += json.usage?.total_tokens || 0;

  const embeddings = json.data
    .sort((a, b) => a.index - b.index)
    .map((d) => d.embedding);

  for (let i = 0; i < embeddings.length; i++) {
    const normalised = l2normalize(embeddings[i]);
    const offset = (start + i) * DIMENSIONS;
    allVectors.set(normalised, offset);
  }

  console.log(`done (${json.usage?.total_tokens ?? "?"} tokens)`);
}

// ── Build index ───────────────────────────────────────────────────────────────

// maxRowByLevel[N] = number of rows whose level <= N
// rows are level-sorted, so rows 0..maxRowByLevel[N]-1 cover all levels 1..N
const maxRowByLevel = {};
{
  let i = 0;
  for (let n = 1; n <= 60; n++) {
    while (i < items.length && items[i].level <= n) i++;
    maxRowByLevel[n] = i;
  }
}

const index = {
  meta: {
    model: MODEL,
    dimensions: DIMENSIONS,
    count: items.length,
    normalized: true,
    created_at: new Date().toISOString(),
    total_tokens_used: totalTokens,
  },
  maxRowByLevel,
  items: items.map((item, row) => ({ row, ...item })),
};

// ── Write files ───────────────────────────────────────────────────────────────

const binPath = join(ROOT, "vocab_embeddings.bin");
const idxPath = join(ROOT, "vocab_embeddings_index.json");

writeFileSync(binPath, Buffer.from(allVectors.buffer));
writeFileSync(idxPath, JSON.stringify(index));

const binMB = (allVectors.buffer.byteLength / 1024 / 1024).toFixed(1);
console.log(`\n✓ ${binPath} (${binMB} MB)`);
console.log(`✓ ${idxPath} (${items.length} items)`);
console.log(`  Total tokens used: ${totalTokens.toLocaleString()}`);
