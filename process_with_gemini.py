"""
process_with_gemini.py – Turn raw scraped articles into QLoRA training data.

Reads JSON files produced by scraper.py from raw_articles/, then for each
article:

  1. Calls Gemini to produce a cleaned plain-text version.
  2. Splits the cleaned text into chunks and calls Gemini again to generate
     Q&A pairs in Mao's rhetorical style.

Outputs training_data.jsonl with two record types:

  • Text-completion record  {"type":"text", "text":"...", "source_url":"...", "title":"..."}
  • Instruction record      {"type":"qa",   "instruction":"...", "input":"",
                             "output":"...", "source_url":"...", "title":"..."}

Requires environment variable GEMINI_API_KEY.
"""

import glob
import json
import os
import re
import sys
import time

from google import genai

# ── Configuration ─────────────────────────────────────────────────────────────

RAW_DIR = "raw_articles"
OUTPUT_FILE = "training_data.jsonl"

GEMINI_MODEL = "gemini-3.1-flash-live-preview"
CHUNK_SIZE = 2500        # characters per text chunk sent to Gemini for cleaning
QA_TEXT_LIMIT = 2000    # characters of cleaned text sent to Gemini for Q&A generation
QA_PAIRS_PER_CHUNK = 5   # number of Q&A pairs to request per chunk
REQUEST_DELAY = 5        # seconds between Gemini calls (free-tier rate-limit)
MAX_RETRIES = 4
MIN_CONTENT_LEN = 150    # skip articles shorter than this

CLEAN_PROMPT = """\
You are processing a historical Chinese political text scraped from a webpage.
The text may contain web navigation elements, repeated headers/footers, and
other artefacts. Your task:

1. Remove navigation text, "Previous / Next article" links, copyright notices,
   site headers, and footers.
2. Keep ALL the actual article content intact.
3. Return ONLY the cleaned text — no explanation, no preamble.

TEXT:
{text}
"""

QA_PROMPT = """\
You are building a dataset to fine-tune a language model in the style of
Mao Zedong.  Based on the excerpt below, generate exactly {n} diverse
question-and-answer pairs.

Guidelines:
- Questions should probe political theory, revolutionary strategy, historical
  analysis, mass-line philosophy, or social criticism as found in the text.
- Answers MUST be written in Mao Zedong's voice: direct, dialectical,
  confident, rich with Marxist-Leninist terminology, and grounded in the
  Chinese revolutionary experience.
- Each answer should be 3–6 sentences minimum.
- Mix Chinese and English naturally if it fits the content.

Return a JSON array and NOTHING else (no markdown fences, no prose).
Each element must have exactly two keys: "instruction" and "output".

EXCERPT:
{text}
"""

# ── Gemini helpers ─────────────────────────────────────────────────────────────

def init_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def call_gemini(prompt: str, client: genai.Client) -> str | None:
    """Call Gemini with *prompt*, retrying on transient errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return response.text
        except Exception as exc:
            print(
                f"  [Gemini attempt {attempt}/{MAX_RETRIES}] Error: {exc}",
                file=sys.stderr,
            )
            if attempt < MAX_RETRIES:
                time.sleep(10 * attempt)
    return None


# ── Text helpers ───────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split *text* into chunks of at most *size* characters, preferring
    paragraph boundaries."""
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current_len + len(para) > size and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def parse_qa_json(raw: str) -> list[dict]:
    """Extract a JSON array from *raw* (which may have markdown fences)."""
    # Strip optional ```json … ``` fences
    raw = raw.strip()
    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    raw = raw.strip()

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Attempt to extract the first JSON array with a regex
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return []


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = init_client(api_key)

    article_files = sorted(
        f for f in glob.glob(os.path.join(RAW_DIR, "*.json"))
        if not f.endswith("manifest.json")
    )
    if not article_files:
        print(f"No article JSON files found in {RAW_DIR}/", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(article_files)} articles with model {GEMINI_MODEL}…\n")

    records: list[dict] = []

    for idx, filepath in enumerate(article_files, 1):
        with open(filepath, "r", encoding="utf-8") as fh:
            article = json.load(fh)

        title = article.get("title", "")
        url = article.get("url", "")
        content = article.get("content", "")

        if not content or len(content) < MIN_CONTENT_LEN:
            print(f"[{idx}/{len(article_files)}] SKIP (too short): {title or filepath}")
            continue

        print(f"[{idx}/{len(article_files)}] {title or url}")

        chunks = chunk_text(content)
        for chunk_idx, chunk in enumerate(chunks, 1):
            # ── Step 1: Clean the chunk ───────────────────────────────────
            clean_resp = call_gemini(CLEAN_PROMPT.format(text=chunk), client)
            time.sleep(REQUEST_DELAY)

            if not clean_resp or len(clean_resp.strip()) < 50:
                print(f"  chunk {chunk_idx}: cleaning returned empty, skipping")
                continue

            cleaned = clean_resp.strip()

            # Emit a text-completion record
            records.append({
                "type": "text",
                "text": cleaned,
                "source_url": url,
                "title": title,
            })

            # ── Step 2: Generate Q&A pairs ────────────────────────────────
            qa_resp = call_gemini(
                QA_PROMPT.format(text=cleaned[:QA_TEXT_LIMIT], n=QA_PAIRS_PER_CHUNK),
                client,
            )
            time.sleep(REQUEST_DELAY)

            if qa_resp:
                qa_pairs = parse_qa_json(qa_resp)
                added = 0
                for pair in qa_pairs:
                    instruction = pair.get("instruction", "").strip()
                    output = pair.get("output", "").strip()
                    if instruction and output:
                        records.append({
                            "type": "qa",
                            "instruction": instruction,
                            "input": "",
                            "output": output,
                            "source_url": url,
                            "title": title,
                        })
                        added += 1
                print(f"  chunk {chunk_idx}/{len(chunks)}: +1 text, +{added} Q&A")
            else:
                print(f"  chunk {chunk_idx}/{len(chunks)}: +1 text, Q&A failed")

    # ── Write JSONL ────────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    text_count = sum(1 for r in records if r["type"] == "text")
    qa_count = sum(1 for r in records if r["type"] == "qa")
    print(
        f"\nWrote {len(records)} records to {OUTPUT_FILE} "
        f"({text_count} text, {qa_count} Q&A)"
    )


if __name__ == "__main__":
    main()
