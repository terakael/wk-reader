#!/usr/bin/env node
/**
 * One-time build script: embeds all WaniKani vocab items and writes:
 *   vocab_embeddings.bin          — Float32Array, L2-normalised, sorted by level (row-major)
 *   vocab_embeddings_index.json   — metadata, per-item index, maxRowByLevel lookup
 *
 * Usage:
 *   node scripts/embed-vocab.js
 *
 * Reads embedding task config from config.json (tasks.embedding + the referenced provider).
 */

import { readFileSync, writeFileSync, existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { createEmbeddingAdapter } from "../adapters/index.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT      = join(__dirname, "..");

const configPath = join(ROOT, "config.json");
if (!existsSync(configPath)) {
  console.error("config.json not found. Copy config.example.json and fill in your values.");
  process.exit(1);
}
const cfg = JSON.parse(readFileSync(configPath, "utf8"));

const taskCfg    = cfg.tasks?.embedding;
const MODEL      = taskCfg?.model;
const DIMENSIONS = taskCfg?.dimensions;

if (!MODEL)      throw new Error("config.json: tasks.embedding.model is required");
if (!DIMENSIONS) throw new Error("config.json: tasks.embedding.dimensions is required");

const embedAdapter = createEmbeddingAdapter(cfg, "embedding");

const BATCH_SIZE = 100;

// ── Load + sort vocab ─────────────────────────────────────────────────────────

const vocabByLevel = JSON.parse(readFileSync(join(ROOT, "vocabulary.json"), "utf8"));

const items = [];
for (let level = 1; level <= 60; level++) {
  for (const item of vocabByLevel[String(level)] || []) {
    items.push({
      character:            item.character,
      reading:              item.reading              || "",
      primary_meaning:      item.primary_meaning      || item.meaning || "",
      alternative_meanings: item.alternative_meanings || [],
      level,
      url: item.url || "",
    });
  }
}
// Sort ascending by level so rows 0..maxRowByLevel[N] covers all levels ≤ N.
items.sort((a, b) => a.level - b.level);

console.log(`Loaded ${items.length} vocab items across 60 levels.`);

// ── Build embedding text ──────────────────────────────────────────────────────

function embeddingText(item) {
  const alts     = item.alternative_meanings.slice(0, 4).join(", ");
  const meanings = alts ? `${item.primary_meaning}, ${alts}` : item.primary_meaning;
  return `Word: ${item.character}\nReading: ${item.reading}\nMeanings: ${meanings}`;
}

// ── Main ──────────────────────────────────────────────────────────────────────

const allVectors = new Float32Array(items.length * DIMENSIONS);

const batches = Math.ceil(items.length / BATCH_SIZE);
console.log(`Embedding in ${batches} batches of up to ${BATCH_SIZE}…\n`);

for (let b = 0; b < batches; b++) {
  const start = b * BATCH_SIZE;
  const end   = Math.min(start + BATCH_SIZE, items.length);
  const texts = items.slice(start, end).map(embeddingText);

  process.stdout.write(`  Batch ${b + 1}/${batches} (items ${start}–${end - 1})… `);

  const vecs = await embedAdapter.embedBatch(texts);

  for (let i = 0; i < vecs.length; i++) {
    allVectors.set(vecs[i], (start + i) * DIMENSIONS);
  }

  console.log("done");
}

// ── Build index ───────────────────────────────────────────────────────────────

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
    model:      MODEL,
    dimensions: DIMENSIONS,
    count:      items.length,
    normalized: true,
    created_at: new Date().toISOString(),
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
