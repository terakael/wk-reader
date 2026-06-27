#!/usr/bin/env python3
"""
anki_sync.py — downloads Anki collection from AnkiWeb and extracts
a vocabulary comfort score map to JSON.

Output: { "化ける": 0.62, "学校": 0.95, ... }

  comfort = ease * (1 - lapse_penalty) * (1 - again_penalty)

  ease         = factor / 2500        (1.0 = default 250% ease)
  lapse_penalty = min(lapses / 10, 0.5)
  again_penalty = min(all_time_again_rate * 2, 0.4)

Only cards with type=2 (actually reviewed) are included.
Cards with the same word but multiple card types (meaning + reading)
are merged by taking the minimum comfort score — the harder direction
drives the score.

Usage:
    ANKIWEB_USERNAME=you@example.com ANKIWEB_PASSWORD=... python3 anki_sync.py
    python3 anki_sync.py --out vocab_comfort.json

k8s: run as a CronJob, write output to a ConfigMap or shared volume.
"""

import argparse
import json
import os
import pathlib
import re
import tempfile
import time

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

WORD_RE = re.compile(r'<div style="font-size: 24px;[^>]*>([^<]+)</div>')
LEVEL_RE = re.compile(r'\blevel-(\d+)\b')

# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------

def sync_collection(username: str, password: str, col_path: str) -> None:
    from anki.collection import Collection
    from anki.sync import SyncAuth
    from anki.sync_pb2 import SyncCollectionResponse

    col = Collection(col_path)
    try:
        auth = col.sync_login(username=username, password=password, endpoint=None)
        result = col.sync_collection(auth=auth, sync_media=False)
        required = result.required

        if required in (SyncCollectionResponse.FULL_SYNC,
                        SyncCollectionResponse.FULL_DOWNLOAD):
            full_auth = SyncAuth(
                hkey=auth.hkey,
                endpoint=result.new_endpoint or None,
            )
            col.close_for_full_sync()
            col.full_upload_or_download(
                auth=full_auth,
                server_usn=result.server_media_usn or None,
                upload=False,
            )
            col.reopen(after_full_sync=True)

        elif required == SyncCollectionResponse.NORMAL_SYNC:
            pass  # incremental sync already applied

        # NO_CHANGES: collection already current, nothing to do

    finally:
        if col.db:
            col.close()


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

EXTRACT_SQL = """
SELECT
    n.flds                                          AS flds,
    n.tags                                          AS tags,
    c.factor                                        AS factor,
    c.lapses                                        AS lapses,
    c.type                                          AS card_type,
    COUNT(r.id)                                     AS total_reviews,
    SUM(CASE WHEN r.ease = 1 THEN 1 ELSE 0 END)    AS again_count
FROM cards c
JOIN notes n ON n.id = c.nid
LEFT JOIN revlog r ON r.cid = c.id
WHERE c.type = 2
  AND n.tags LIKE '%wanikani%'
GROUP BY c.id
"""


def extract_word(flds: str) -> str | None:
    """Extract the Japanese word from the 24px div in the front field."""
    front = flds.split("\x1f")[0]
    m = WORD_RE.search(front)
    return m.group(1).strip() if m else None


def compute_comfort(factor: int, lapses: int, total_reviews: int, again_count: int) -> float:
    ease = min(factor / 2500.0, 1.0)  # cap at 1.0; above-average ease doesn't add extra comfort
    lapse_penalty = min(lapses / 10.0, 0.5)
    again_rate = again_count / total_reviews if total_reviews > 0 else 0.0
    again_penalty = min(again_rate * 2.0, 0.4)
    return round(ease * (1.0 - lapse_penalty) * (1.0 - again_penalty), 4)


def extract_comfort_map(col_path: str) -> dict[str, float]:
    from anki.collection import Collection

    col = Collection(col_path)
    try:
        rows = col.db.all(EXTRACT_SQL)
    finally:
        col.close()

    # word → min comfort (multiple card types per word — use the weaker direction)
    scores: dict[str, float] = {}
    for flds, tags, factor, lapses, card_type, total_reviews, again_count in rows:
        word = extract_word(flds)
        if not word:
            continue
        comfort = compute_comfort(factor, lapses, total_reviews, again_count)
        if word not in scores or comfort < scores[word]:
            scores[word] = comfort

    return scores


# ---------------------------------------------------------------------------
# metadata (level + card kind per word, useful for filtering)
# ---------------------------------------------------------------------------

METADATA_SQL = """
SELECT DISTINCT
    n.flds  AS flds,
    n.tags  AS tags
FROM cards c
JOIN notes n ON n.id = c.nid
WHERE c.type = 2
  AND n.tags LIKE '%wanikani%'
"""


def extract_metadata(col_path: str) -> dict[str, dict]:
    """Returns { word: { level: int, kind: str } } for reviewed cards."""
    from anki.collection import Collection

    col = Collection(col_path)
    try:
        rows = col.db.all(METADATA_SQL)
    finally:
        col.close()

    meta: dict[str, dict] = {}
    for flds, tags in rows:
        word = extract_word(flds)
        if not word:
            continue
        lm = LEVEL_RE.search(tags)
        level = int(lm.group(1)) if lm else 0
        tag_set = set(tags.split())
        if "radical" in tag_set:
            kind = "radical"
        elif any(t.startswith("kanji") for t in tag_set):
            kind = "kanji"
        elif any(t.startswith("vocab") for t in tag_set):
            kind = "vocabulary"
        else:
            kind = "unknown"
        # keep the first occurrence (multiple card types share the same word+level)
        if word not in meta:
            meta[word] = {"level": level, "kind": kind}

    return meta


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def load_env() -> None:
    env_path = pathlib.Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    load_env()

    parser = argparse.ArgumentParser(description="Sync Anki and export comfort scores")
    parser.add_argument("--out", default="vocab_comfort.json",
                        help="Output JSON path (default: vocab_comfort.json)")
    parser.add_argument("--with-metadata", action="store_true",
                        help="Include level + kind in output alongside comfort score")
    parser.add_argument("--skip-sync", action="store_true",
                        help="Skip AnkiWeb sync, use existing collection")
    parser.add_argument("--col", default=None,
                        help="Path to existing collection.anki2 (implies --skip-sync)")
    args = parser.parse_args()

    username = os.environ.get("ANKIWEB_USERNAME", "")
    password = os.environ.get("ANKIWEB_PASSWORD", "")

    if args.col:
        col_path = args.col
        skip_sync = True
    else:
        tmp_dir = tempfile.mkdtemp(prefix="anki_sync_")
        col_path = os.path.join(tmp_dir, "collection.anki2")
        skip_sync = args.skip_sync

    if not skip_sync:
        if not username or not password:
            raise SystemExit("ANKIWEB_USERNAME and ANKIWEB_PASSWORD must be set")
        print(f"Syncing from AnkiWeb...", flush=True)
        t0 = time.time()
        sync_collection(username, password, col_path)
        print(f"Sync complete ({time.time() - t0:.1f}s)", flush=True)

    print("Extracting comfort scores...", flush=True)
    scores = extract_comfort_map(col_path)

    if args.with_metadata:
        meta = extract_metadata(col_path)
        output = {
            word: {"comfort": score, **meta.get(word, {})}
            for word, score in scores.items()
        }
    else:
        output = scores

    out_path = pathlib.Path(args.out)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Wrote {len(scores)} words → {out_path}", flush=True)


if __name__ == "__main__":
    main()
