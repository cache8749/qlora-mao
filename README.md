# qlora-mao

Automated pipeline that scrapes Mao Zedong's writings from
[marxists.org](https://www.marxists.org/chinese/maozedong/index.htm)
(public domain), processes them with the Gemini API, and publishes a
`training_data.jsonl` file to GitHub Releases for use in QLoRA fine-tuning.

---

## How it works

```
marxists.org  ──scraper.py──►  raw_articles/*.json
                                       │
                         process_with_gemini.py
                                       │
                               training_data.jsonl
                                       │
                          GitHub Actions Release upload
```

### Step 1 – Scrape (`scraper.py`)

Crawls both index pages:

- `https://www.marxists.org/chinese/maozedong/index.htm`
- `https://www.marxists.org/chinese/maozedong/1968/index.htm`

Follows every article link found and saves each article as a JSON file
inside `raw_articles/`.  A `raw_articles/manifest.json` is also written with
the article list.

### Step 2 – Process (`process_with_gemini.py`)

For each article the script:

1. Calls **Gemini 3.1 Flash Live Preview** to remove web-navigation artefacts and produce
   clean plain text.
2. Splits the cleaned text into ~2 500-character chunks and calls Gemini
   again to generate five Q&A pairs per chunk in Mao's rhetorical style.

Output is `training_data.jsonl` with two record types:

```jsonc
// Text-completion record (causal LM pre-training)
{"type":"text","text":"…","source_url":"…","title":"…"}

// Instruction record (instruction fine-tuning / Alpaca format)
{"type":"qa","instruction":"…","input":"","output":"…","source_url":"…","title":"…"}
```

### Step 3 – GitHub Actions (`pipeline.yml`)

The workflow runs:

- **Manually** via *Actions → Run workflow*
- **Automatically** on the 1st of each month at 02:00 UTC

It has two jobs:

| Job | What it does |
|-----|-------------|
| `scrape` | Runs `scraper.py`, uploads `raw_articles/` as a workflow artifact |
| `process` | Downloads the artifact, runs `process_with_gemini.py`, publishes `training_data.jsonl` to a new GitHub Release |

---

## Setup

1. **Fork / clone** this repository.
2. Go to *Settings → Secrets and variables → Actions* and add a secret
   named **`GEMINI_API_KEY`** with your Google AI Studio API key.
3. Trigger the workflow manually or wait for the scheduled run.

The generated `training_data.jsonl` will appear in the repository's
**Releases** section.

---

## Running locally

```bash
pip install -r requirements.txt

# Scrape
python scraper.py          # writes raw_articles/

# Process (requires API key)
export GEMINI_API_KEY=<your-key>
python process_with_gemini.py   # writes training_data.jsonl
```

---

## License

The scraper and processing code in this repository is released under the
[MIT License](LICENSE).  The scraped content is in the public domain.