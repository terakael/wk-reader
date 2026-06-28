#!/usr/bin/env node
/**
 * build_dict.js — downloads the full JMdict (English) from jmdict-simplified
 * and builds dict.sqlite for use by server.js.
 *
 * Run once (or whenever you want to update the dictionary):
 *   node scripts/build_dict.js
 *
 * The downloaded JSON is kept as jmdict-full.json so re-runs skip the download.
 * Delete it to force a fresh download.
 */

import { execSync }                   from "child_process";
import { existsSync, unlinkSync }     from "fs";
import { readFile }                   from "fs/promises";
import { join, dirname }              from "path";
import { fileURLToPath }              from "url";
import Database                       from "better-sqlite3";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT      = join(__dirname, "..");

// ── Download ─────────────────────────────────────────────────────────────────

const RELEASE   = "3.6.2+20260622163854";
const ASSET     = `jmdict-eng-${RELEASE}.json.tgz`;
const URL       = `https://github.com/scriptin/jmdict-simplified/releases/download/${encodeURIComponent(RELEASE)}/${ASSET}`;
const JSON_FILE = "jmdict-eng-3.6.2.json";   // name inside the archive
const JSON_PATH = join(ROOT, JSON_FILE);
const DB_PATH   = join(ROOT, "dict.sqlite");

if (!existsSync(JSON_PATH)) {
  console.log("Downloading full JMdict (~11 MB compressed)...");
  execSync(`curl -sL "${URL}" | tar -xz -C "${ROOT}"`, { stdio: "inherit" });
  console.log("Download complete.");
} else {
  console.log(`Found ${JSON_FILE}, skipping download.`);
}

// ── Build ─────────────────────────────────────────────────────────────────────

console.log("Parsing JSON...");
const { words, tags } = JSON.parse(await readFile(JSON_PATH, "utf8"));
console.log(`  ${words.length} entries`);

if (existsSync(DB_PATH)) {
  console.log("Removing existing dict.sqlite...");
  unlinkSync(DB_PATH);
}

const db = new Database(DB_PATH);

db.exec(`
  CREATE TABLE dict (
    word    TEXT PRIMARY KEY,
    reading TEXT NOT NULL,
    meanings TEXT NOT NULL,   -- JSON array of up to 3 English glosses
    pos      TEXT NOT NULL    -- JSON array of up to 2 human-readable POS strings
  );
`);

const insert = db.prepare("INSERT OR IGNORE INTO dict VALUES (?, ?, ?, ?)");

let rows = 0;

const insertAll = db.transaction(() => {
  for (const entry of words) {
    const reading  = entry.kana.find(k => k.common)?.text ?? entry.kana[0]?.text ?? "";
    const sense    = entry.sense[0] ?? {};
    const meanings = JSON.stringify((sense.gloss ?? []).slice(0, 3).map(g => g.text));
    const pos      = JSON.stringify((sense.partOfSpeech ?? []).slice(0, 2).map(p => tags[p] ?? p));

    for (const k of entry.kanji) { insert.run(k.text, reading, meanings, pos); rows++; }
    for (const k of entry.kana)  { insert.run(k.text, reading, meanings, pos); rows++; }
  }
});

console.log("Inserting rows...");
insertAll();
db.close();

console.log(`Done — ${rows} rows written to dict.sqlite.`);
