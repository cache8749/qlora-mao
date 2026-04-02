"""
scraper.py – Crawl Mao Zedong's writings from marxists.org
and save each article as a JSON file under raw_articles/.

Index pages scraped:
  https://www.marxists.org/chinese/maozedong/index.htm
  https://www.marxists.org/chinese/maozedong/1968/index.htm
"""

import json
import os
import sys
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ── Configuration ────────────────────────────────────────────────────────────

INDEX_URLS = [
    "https://www.marxists.org/chinese/maozedong/index.htm",
    "https://www.marxists.org/chinese/maozedong/1968/index.htm",
]

OUTPUT_DIR = "raw_articles"
DELAY = 1.5          # polite delay between HTTP requests (seconds)
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; academic-research-bot; "
        "+https://github.com/cache8749/qlora-mao)"
    )
}

# Only follow links that stay within the Mao section of marxists.org
ALLOWED_PATH_PREFIX = "/chinese/maozedong"


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def get_page(url: str) -> str | None:
    """Fetch *url* and return the decoded HTML string, or None on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url, headers=HEADERS, timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            # marxists.org pages may be GB2312 encoded; fall back to UTF-8
            enc = response.apparent_encoding or "utf-8"
            return response.content.decode(enc, errors="replace")
        except Exception as exc:
            print(
                f"  [attempt {attempt}/{MAX_RETRIES}] Failed to fetch {url}: {exc}",
                file=sys.stderr,
            )
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
    return None


# ── Link extraction ───────────────────────────────────────────────────────────

def extract_article_links(html: str, index_url: str) -> list[str]:
    """
    Return a de-duplicated list of article URLs found in *html*.

    Rules:
    - Must be within ALLOWED_PATH_PREFIX on marxists.org
    - Must end with .htm or .html (i.e., actual page, not a directory)
    - Must NOT be an index page itself
    - Anchors, mailto links, and off-site links are ignored
    """
    soup = BeautifulSoup(html, "lxml")
    seen: dict[str, None] = {}

    for anchor in soup.find_all("a", href=True):
        href: str = anchor["href"].strip()

        # Skip anchors and mail links
        if href.startswith("#") or href.startswith("mailto:"):
            continue

        full_url = urljoin(index_url, href).split("#")[0]  # drop fragment

        # Must be on marxists.org within the allowed sub-path
        if "marxists.org" not in full_url:
            continue
        from urllib.parse import urlparse
        path = urlparse(full_url).path
        if not path.startswith(ALLOWED_PATH_PREFIX):
            continue
        if not (path.endswith(".htm") or path.endswith(".html")):
            continue
        if path.endswith("index.htm") or path.endswith("index.html"):
            continue

        seen[full_url] = None

    return list(seen)


# ── Content extraction ────────────────────────────────────────────────────────

def extract_content(html: str, url: str) -> dict:
    """
    Parse article HTML and return a dict with keys:
    url, title, content (paragraphs joined by double newlines).
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove clutter tags
    for tag in soup.find_all(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # Title: prefer <h1>, fall back to <title>
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(separator=" ", strip=True)
    if not title and soup.title:
        title = soup.title.get_text(strip=True)

    # Gather text from all <p> tags with meaningful content
    paragraphs = [
        p.get_text(separator=" ", strip=True)
        for p in soup.find_all("p")
        if len(p.get_text(strip=True)) > 20
    ]

    # Fall back to full body text if no paragraphs found
    if not paragraphs:
        body = soup.find("body")
        if body:
            paragraphs = [
                line.strip()
                for line in body.get_text(separator="\n").splitlines()
                if len(line.strip()) > 20
            ]

    content = "\n\n".join(paragraphs)
    return {"url": url, "title": title, "content": content}


# ── URL → filename ────────────────────────────────────────────────────────────

def url_to_filename(url: str) -> str:
    """Convert a URL to a safe JSON filename."""
    path = url.replace("https://", "").replace("http://", "")
    safe = path.replace("/", "__").replace(".", "_")
    return safe + ".json"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Collect all unique article URLs from both index pages
    all_urls: dict[str, None] = {}
    for index_url in INDEX_URLS:
        print(f"Fetching index: {index_url}")
        html = get_page(index_url)
        if html is None:
            print(f"  ERROR: could not fetch index {index_url}", file=sys.stderr)
            continue
        links = extract_article_links(html, index_url)
        print(f"  Found {len(links)} article links")
        for link in links:
            all_urls[link] = None
        time.sleep(DELAY)

    article_urls = list(all_urls)
    print(f"\nTotal unique articles to scrape: {len(article_urls)}\n")

    # 2. Scrape each article
    articles: list[dict] = []
    for i, url in enumerate(article_urls, 1):
        print(f"[{i}/{len(article_urls)}] {url}")
        html = get_page(url)
        if html is None:
            continue

        article = extract_content(html, url)
        if not article["content"]:
            print(f"  WARNING: no content extracted, skipping")
            continue

        articles.append(article)

        filename = url_to_filename(url)
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(article, fh, ensure_ascii=False, indent=2)

        time.sleep(DELAY)

    # 3. Write manifest
    manifest = {
        "total": len(articles),
        "articles": [{"url": a["url"], "title": a["title"]} for a in articles],
    }
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print(f"\nDone. Scraped {len(articles)} articles → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
