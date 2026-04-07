"""
process_local.py – Turn raw scraped articles into QLoRA training data using a
local llama.cpp server (http://localhost:8080/completion).

Reads JSON files from raw_articles/, then for each article:
  1. Calls the local model to produce a cleaned plain-text version.
  2. Splits the cleaned text into chunks and calls the model again to generate
     Q&A pairs in Mao's rhetorical style.

Outputs training_data.jsonl with two record types:
  • CPT record   {"text": "…"}
  • SFT record   {"messages": [...]}
"""

import glob
import json
import os
import re
import sys

import requests

# ── Configuration ─────────────────────────────────────────────────────────────

RAW_DIR = "raw_articles"
OUTPUT_FILE = "training_data.jsonl"

LLAMA_URL = "http://localhost:8080/v1/chat/completions"
LLAMA_MODEL = "Qwen3.5-9B-UD-Q4_K_XL"
N_PREDICT = 2048

CHUNK_SIZE = 8000         # characters per chunk (keep within context window)
QA_TEXT_LIMIT = 6000      # characters of cleaned text sent for Q&A generation
QA_PAIRS_PER_CHUNK = 5
MAX_RETRIES = 5
MIN_CONTENT_LEN = 150

SYSTEM_PROMPT = (
    "你是毛泽东，中国共产党主席，中华人民共和国的缔造者。"
    "请用第一人称以中文回答，语气权威、辩证清晰、充满革命信念。"
    "以马克思列宁主义理论和中国革命的具体实践为基础，直接而毫不妥协。"
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
你正在构建一个用于微调语言模型的数据集，使其模仿毛泽东的风格。
请根据以下摘录，生成恰好 {n} 个多样化的问答对。

要求：
- 问题须以中文提出，涉及政治理论、革命战略、历史分析、群众路线哲学或社会批评等主题，内容须源自摘录。
- 答案必须以毛泽东的口吻用中文作答：直接、辩证、自信，富含马克思列宁主义术语，根植于中国革命的具体经验。
- 每个答案最少包含3至6句话。
- 问题和答案均须使用中文。

只返回一个 JSON 数组，不得包含 Markdown 代码块或任何说明文字。
每个元素必须恰好包含两个键："question" 和 "answer"。

摘录：
{text}
"""

# ── llama.cpp helper ───────────────────────────────────────────────────────────

def call_local(prompt: str) -> str | None:
    """Send *prompt* to the local llama.cpp server and return the generated text.
    Uses streaming to avoid read timeouts on slow CPU inference."""
    payload = {
        "model": LLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": N_PREDICT,
        "temperature": 0.7,
        "stream": True,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.post(LLAMA_URL, json=payload, stream=True, timeout=(10, 600)) as response:
                response.raise_for_status()
                chunks = []
                for line in response.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            delta = json.loads(data)["choices"][0]["delta"]
                            chunks.append(delta.get("content") or "")
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass
                return "".join(chunks).strip()
        except Exception as exc:
            print(f"  [attempt {attempt}/{MAX_RETRIES}] Error: {exc}", file=sys.stderr)

    return None


# ── Text helpers ───────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
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
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    article_files = sorted(
        f for f in glob.glob(os.path.join(RAW_DIR, "*.json"))
        if not f.endswith("manifest.json")
    )
    if not article_files:
        print(f"No article JSON files found in {RAW_DIR}/", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(article_files)} articles via local llama.cpp ({LLAMA_URL})…\n")

    total_cpt = 0
    total_sft = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for idx, filepath in enumerate(article_files, 1):
            with open(filepath, "r", encoding="utf-8") as fh:
                article = json.load(fh)

            title   = article.get("title", "")
            url     = article.get("url", "")
            content = article.get("content", "")

            if not content or len(content) < MIN_CONTENT_LEN:
                print(f"[{idx}/{len(article_files)}] SKIP (too short): {title or filepath}")
                continue

            print(f"[{idx}/{len(article_files)}] {title or url}")

            chunks = chunk_text(content)
            for chunk_idx, chunk in enumerate(chunks, 1):
                # ── Step 1: Clean ─────────────────────────────────────────────
                clean_resp = call_local(CLEAN_PROMPT.format(text=chunk))

                if not clean_resp or len(clean_resp.strip()) < 50:
                    print(f"  chunk {chunk_idx}: cleaning returned empty, skipping")
                    continue

                cleaned = clean_resp.strip()

                out.write(json.dumps({"text": cleaned}, ensure_ascii=False) + "\n")
                out.flush()
                total_cpt += 1

                # ── Step 2: Q&A ───────────────────────────────────────────────
                qa_resp = call_local(
                    QA_PROMPT.format(text=cleaned[:QA_TEXT_LIMIT], n=QA_PAIRS_PER_CHUNK)
                )

                added = 0
                if qa_resp:
                    for pair in parse_qa_json(qa_resp):
                        question = pair.get("question", "").strip()
                        answer   = pair.get("answer", "").strip()
                        if question and answer:
                            out.write(json.dumps(make_sft_record(question, answer), ensure_ascii=False) + "\n")
                            out.flush()
                            total_sft += 1
                            added += 1

                print(f"  chunk {chunk_idx}/{len(chunks)}: +1 CPT, +{added} SFT")

    print(
        f"\nWrote {total_cpt + total_sft} records to {OUTPUT_FILE} "
        f"({total_cpt} CPT, {total_sft} SFT)"
    )


if __name__ == "__main__":
    main()
