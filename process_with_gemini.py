"""
process_with_gemini.py – Turn raw scraped articles into QLoRA training data.

Reads JSON files produced by scraper.py from raw_articles/, then for each
article:

  1. Calls Gemini to produce a cleaned plain-text version.
  2. Splits the cleaned text into chunks and calls Gemini again to generate
     Q&A pairs in Mao's rhetorical style.

Outputs training_data.jsonl with two record types for use with Unsloth:

  • CPT record   {"text": "…"}
      Raw cleaned article text for continued pretraining.

  • SFT record   {"messages": [{"role": "system", …}, {"role": "user", …},
                               {"role": "assistant", …}]}
      ChatML-format conversation for supervised instruction fine-tuning.

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

GEMINI_MODEL = "gemini-2.0-flash"
CHUNK_SIZE = 2500        # characters per text chunk sent to Gemini for cleaning
QA_TEXT_LIMIT = 2000     # characters of cleaned text sent to Gemini for Q&A generation
QA_PAIRS_PER_CHUNK = 5   # number of Q&A pairs to request per chunk
REQUEST_DELAY = 5        # seconds between Gemini calls (free-tier rate-limit)
MAX_RETRIES = 4
MIN_CONTENT_LEN = 150    # skip articles shorter than this

SYSTEM_PROMPT = (
    "You are Mao Zedong, Chairman of the Chinese Communist Party and founder "
    "of the People's Republic of China. Speak in the first person with "
    "authority, dialectical clarity, and revolutionary conviction. Ground your "
    "answers in Marxist-Leninist theory and the concrete experience of the "
    "Chinese revolution. Be direct and uncompromising."
)

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
Each element must have exactly two keys: "question" and "answer".

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

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return []


def make_sft_record(question: str, answer: str) -> dict:
    """Wrap a Q&A pair into a ChatML-format SFT record for Unsloth."""
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


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

    cpt_records: list[dict] = []
    sft_records: list[dict] = []

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

            # Emit a CPT record (plain text, standard Unsloth format)
            cpt_records.append({"text": cleaned})

            # ── Step 2: Generate Q&A pairs ────────────────────────────────
            qa_resp = call_gemini(
                QA_PROMPT.format(text=cleaned[:QA_TEXT_LIMIT], n=QA_PAIRS_PER_CHUNK),
                client,
            )
            time.sleep(REQUEST_DELAY)

            added = 0
            if qa_resp:
                qa_pairs = parse_qa_json(qa_resp)
                for pair in qa_pairs:
                    question = pair.get("question", "").strip()
                    answer = pair.get("answer", "").strip()
                    if question and answer:
                        sft_records.append(make_sft_record(question, answer))
                        added += 1

            print(f"  chunk {chunk_idx}/{len(chunks)}: +1 CPT, +{added} SFT")

    # ── Write JSONL ────────────────────────────────────────────────────────────
    all_records = cpt_records + sft_records
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        for record in all_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"\nWrote {len(all_records)} records to {OUTPUT_FILE} "
        f"({len(cpt_records)} CPT, {len(sft_records)} SFT)"
    )


if __name__ == "__main__":
    main()
