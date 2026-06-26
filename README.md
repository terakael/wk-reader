# wk-reader

A Japanese reading practice tool for [WaniKani](https://www.wanikani.com) learners. Enter a story prompt (or generate a random one), pick your WaniKani level, and get a short Japanese story using only vocabulary you've already learned.

Every content word in the story comes from your known WaniKani vocabulary. Click any highlighted word to see its reading, meaning, and level.

![screenshot](screenshot.png)

---

## How it works

1. Your prompt is embedded and used to retrieve the most semantically relevant words from your known vocabulary
2. A brief prompt expansion grounds the retrieval in concrete scene detail
3. Gemini generates a 10–15 sentence story constrained strictly to the retrieved word list
4. The story is tokenised with kuromoji and annotated client-side:
   - 🟢 **Green** — word is within your level
   - 🟠 **Orange** — word is in WaniKani but above your level
   - 🔴 **Red** — kanji not in WaniKani at all

---

## Setup

### Requirements

- Node.js 22+
- API keys for your chosen embedding and generation providers (see config below)
- The pre-built embedding files (see below)

### Install

```bash
git clone git@github.com:terakael/wk-reader.git
cd wk-reader
npm install
```

### Configure

Copy the example config and fill in your values:

```bash
cp config.example.json config.json
```

Edit `config.json`. The config has two sections:

**`providers`** — named connection definitions. Each has a `type` (`openai` or `gemini`), a `baseUrl`, and credentials. Credentials can be a literal `apiKey` or an `apiKeyCmd` shell command whose stdout is used as the key:

```json
"apiKeyCmd": "security find-generic-password -a api-keys -s openai -w"
```

For `gemini` providers, the default auth header is `x-goog-api-key`. Override with `"authHeader": "Authorization"` if your endpoint expects a Bearer token instead.

**`tasks`** — one entry each for `embedding`, `enhance`, and `story`. Each references a provider by name and specifies the model. The `embedding` task also requires `dimensions`. Optionally add `queryPrefix` and `documentPrefix` to prepend task-instruction strings to your inputs — useful for Gemini Embedding 2's asymmetric retrieval format:

```json
"embedding": {
  "provider": "gemini",
  "model": "gemini-embedding-2",
  "dimensions": 1536,
  "queryPrefix": "task: search result | query: ",
  "documentPrefix": "title: none | text: "
}
```

The three tasks can point at different providers, so you can mix OpenAI embeddings with Gemini generation, or run everything through one provider.

### Embedding files

The vocab embeddings (~41 MB) are not included in the repo. Download and place them in the project root:

- `vocab_embeddings.bin`
- `vocab_embeddings_index.json`

> **Alternatively**, rebuild them yourself (costs ~$0.02 with OpenAI's `text-embedding-3-small`):
> ```bash
> node scripts/embed-vocab.js
> ```
> The script reads your `embedding` task config from `config.json`, so make sure that's set up first.
> Note: if you switch embedding models, the stored vectors and the live query vectors must come from the same model — a mismatch will silently return bad results. Rebuild whenever you change the model or dimensions.

### Run

```bash
node server.js
```

Open [http://localhost:3000](http://localhost:3000).

---

## Tech

- **Backend:** Node.js, Express 5, kuromoji (Japanese tokeniser)
- **Frontend:** Vanilla JS, single HTML file
- **Embeddings:** configurable — OpenAI or Gemini provider, any compatible model
- **Prompt enhancement:** configurable — defaults to `gemini-2.5-flash` (no thinking)
- **Story generation:** configurable — defaults to `gemini-2.5-flash`, streaming SSE (no thinking)
- **Annotation:** greedy longest-match merge over kuromoji tokens → WaniKani vocab lookup
