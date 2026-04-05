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

from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────

RAW_DIR = "raw_articles"
OUTPUT_FILE = "training_data.jsonl"

GEMINI_MODEL = "deepseek-ai/deepseek-v3.2"
FALLBACK_MODEL = "z-ai/glm4.7"
CHUNK_SIZE = 25000        # characters per text chunk sent to Gemini for cleaning
QA_TEXT_LIMIT = 20000     # characters of cleaned text sent to Gemini for Q&A generation
QA_PAIRS_PER_CHUNK = 5   # number of Q&A pairs to request per chunk
MAX_RETRIES = 10
MIN_CONTENT_LEN = 150    # skip articles shorter than this

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

# ── Gemini helpers ─────────────────────────────────────────────────────────────

def init_client(api_key: str) -> OpenAI:
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )


def call_gemini(prompt: str, client: OpenAI) -> str | None:
    """Call the NVIDIA API with *prompt*, retrying on transient errors.
    Falls back to FALLBACK_MODEL if all retries on GEMINI_MODEL fail."""
    for model in dict.fromkeys([GEMINI_MODEL, FALLBACK_MODEL]):  # dedup while preserving order
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=1,
                    top_p=0.95,
                    max_tokens=8192,
                    extra_body={"chat_template_kwargs": {"thinking": True}},
                    stream=True,
                )
                parts: list[str] = []
                for chunk in completion:
                    if not getattr(chunk, "choices", None):
                        continue
                    if chunk.choices[0].delta.content is not None:
                        parts.append(chunk.choices[0].delta.content)
                return "".join(parts)
            except Exception as exc:
                print(
                    f"  [{model} attempt {attempt}/{MAX_RETRIES}] Error: {exc}",
                    file=sys.stderr,
                )
        if model != FALLBACK_MODEL:
            print(f"  Primary model {model} failed, trying fallback {FALLBACK_MODEL}…", file=sys.stderr)
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

    total_cpt = 0
    total_sft = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
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

                if not clean_resp or len(clean_resp.strip()) < 50:
                    print(f"  chunk {chunk_idx}: cleaning returned empty, skipping")
                    continue

                cleaned = clean_resp.strip()

                # Write CPT record immediately
                out.write(json.dumps({"text": cleaned}, ensure_ascii=False) + "\n")
                out.flush()
                total_cpt += 1

                # ── Step 2: Generate Q&A pairs ────────────────────────────────
                qa_resp = call_gemini(
                    QA_PROMPT.format(text=cleaned[:QA_TEXT_LIMIT], n=QA_PAIRS_PER_CHUNK),
                    client,
                )

                added = 0
                if qa_resp:
                    qa_pairs = parse_qa_json(qa_resp)
                    for pair in qa_pairs:
                        question = pair.get("question", "").strip()
                        answer = pair.get("answer", "").strip()
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
