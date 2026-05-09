"""
FOMC Minutes Scraper
====================
Downloads and extracts plain text from FOMC meeting minutes from:
  - https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm  (2021–present)
  - https://www.federalreserve.gov/monetarypolicy/fomchistoricalYYYY.htm (historical, per year)

The historical site uses a TWO-LEVEL structure:
  Level 1: fomc_historical_year.htm  →  lists links like /monetarypolicy/fomchistorical2015.htm
  Level 2: fomchistoricalYYYY.htm    →  contains the actual minutes links (PDF or HTML)

Output:
  fomc_minutes/
    YYYY-MM-DD_minutes.txt   (one clean text file per meeting)
    index.csv                (date, url, local_file, status)

Requirements:
  pip install requests beautifulsoup4 lxml pdfplumber
  OR:
  uv run --with requests --with beautifulsoup4 --with lxml --with pdfplumber python fomc_minutes_scraper.py
"""

import os
import re
import time
import csv
import io
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

try:
    import pdfplumber
    PDF_BACKEND = "pdfplumber"
except ImportError:
    pdfplumber = None
    PDF_BACKEND = None

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL      = "https://www.federalreserve.gov"
CURRENT_URL   = f"{BASE_URL}/monetarypolicy/fomccalendars.htm"

# Level-1 index: lists links like /monetarypolicy/fomchistoricalYYYY.htm
HIST_INDEX_URL = f"{BASE_URL}/monetarypolicy/fomc_historical_year.htm"

OUTPUT_DIR    = Path("fomc_minutes")
INDEX_FILE    = OUTPUT_DIR / "index.csv"

REQUEST_DELAY = 1.2   # seconds between requests (be polite)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; FOMC-Research-Scraper/2.0)"
    )
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Check dependencies ────────────────────────────────────────────────────────

def check_dependencies():
    if pdfplumber is None:
        log.error("pdfplumber is not installed.")
        log.error("Install it with:")
        log.error("  pip install pdfplumber")
        log.error("  OR use the uv command below:")
        log.error("  uv run --with requests --with beautifulsoup4 --with lxml --with pdfplumber python fomc_minutes_scraper.py")
        raise SystemExit(1)
    log.info(f"PDF backend: {PDF_BACKEND}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return BeautifulSoup(resp.text, "lxml")


def fetch_bytes(url: str) -> tuple:
    """Return (content_bytes, content_type)."""
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return resp.content, resp.headers.get("Content-Type", "")


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF given its raw bytes."""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        return "\n\n".join(pages)


def extract_text_from_html(html_bytes: bytes) -> str:
    """Strip HTML tags and return clean text."""
    soup = BeautifulSoup(html_bytes, "lxml")

    # Try the Fed's content containers first
    for selector in [
        "div#article",
        "div.col-xs-12.col-sm-8.col-md-8",
        "div#content",
        "div.text",
        "article",
        "main",
    ]:
        container = soup.select_one(selector)
        if container:
            return container.get_text(separator="\n").strip()

    # Fallback: whole body
    return soup.get_text(separator="\n").strip()


def fetch_and_extract(url: str) -> str:
    """Download a URL and return extracted plain text regardless of format."""
    raw_bytes, content_type = fetch_bytes(url)
    ct = content_type.lower()

    if "pdf" in ct or url.lower().endswith(".pdf"):
        log.info("    -> PDF detected, extracting text...")
        return extract_text_from_pdf(raw_bytes)
    elif "text/plain" in ct:
        return raw_bytes.decode("utf-8", errors="replace")
    else:
        # HTML or unknown — treat as HTML
        return extract_text_from_html(raw_bytes)


def clean_text(text: str) -> str:
    """Basic cleanup: collapse excessive blank lines."""
    lines = text.splitlines()
    cleaned = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned.append("")
        else:
            blank_count = 0
            cleaned.append(line)
    return "\n".join(cleaned).strip()


def infer_date_from_url(url: str) -> str:
    m = re.search(r"(\d{8})", url)
    if m:
        raw = m.group(1)
        try:
            dt = datetime.strptime(raw, "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return raw
    return "unknown"


def safe_filename(date_str: str) -> str:
    return f"{date_str}_minutes.txt"


# ── Link discovery ─────────────────────────────────────────────────────────────

def find_minutes_links_on_page(soup: BeautifulSoup, page_url: str) -> list:
    """
    Find all links that point to actual minutes documents (PDF or HTML).
    Minutes links on per-year pages typically look like:
      /monetarypolicy/fomcminutes20150128.pdf
      /monetarypolicy/fomcminutes20150128.htm
    On the current calendar they look like:
      /monetarypolicy/fomcminutes20230201.htm
    """
    results = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        href_lower = href.lower()
        text_lower = a.get_text(strip=True).lower()

        is_minutes = (
            "fomcminutes" in href_lower          # direct minutes file URL
            or "minutes" in text_lower            # link text says "minutes"
            or bool(re.search(r"minutes\d{8}", href_lower))
        )

        if not is_minutes:
            continue

        # Exclude links that point to another year-index page (not actual minutes)
        if "fomchistorical" in href_lower and "minutes" not in href_lower:
            continue

        full_url = urljoin(page_url, href)
        if full_url in seen:
            continue
        seen.add(full_url)

        date_str = infer_date_from_url(full_url)
        results.append({"date": date_str, "url": full_url})

    return results


def discover_historical_year_urls(soup: BeautifulSoup) -> list:
    """
    From the fomc_historical_year.htm index page, collect all per-year page URLs.

    The Fed uses hrefs like:
        /monetarypolicy/fomchistorical2015.htm
    NOT query-string style ?year=2015.
    """
    year_urls = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Match pattern: fomchistoricalYYYY.htm  (4-digit year, no "minutes" in it)
        if re.search(r"fomchistorical\d{4}\.htm", href, re.IGNORECASE):
            full = urljoin(HIST_INDEX_URL, href)
            if full not in seen:
                seen.add(full)
                year_urls.append(full)

    return sorted(year_urls)


def discover_all_links() -> list:
    all_links = []

    # ── 1. Current / recent meetings (2021–present) ──────────────────────────
    log.info("Fetching current calendar: %s", CURRENT_URL)
    soup = get_soup(CURRENT_URL)
    links = find_minutes_links_on_page(soup, CURRENT_URL)
    log.info("  Found %d minutes links on current calendar.", len(links))
    all_links.extend(links)

    # ── 2. Historical pages (1936–2020) via two-level navigation ─────────────
    #   Level 1: fomc_historical_year.htm  → lists fomchistoricalYYYY.htm pages
    #   Level 2: fomchistoricalYYYY.htm    → contains actual minutes links
    log.info("Fetching historical index: %s", HIST_INDEX_URL)
    soup = get_soup(HIST_INDEX_URL)

    year_urls = discover_historical_year_urls(soup)
    log.info("  Found %d historical year pages.", len(year_urls))

    for year_url in year_urls:
        m = re.search(r"fomchistorical(\d{4})\.htm", year_url, re.IGNORECASE)
        year_label = m.group(1) if m else year_url
        log.info("  Scraping year %s ...", year_label)
        try:
            soup = get_soup(year_url)
            links = find_minutes_links_on_page(soup, year_url)
            log.info("    -> %d minutes links", len(links))
            all_links.extend(links)
        except Exception as exc:
            log.warning("    Could not fetch %s: %s", year_url, exc)

    # ── Deduplicate by URL and sort chronologically ───────────────────────────
    seen_urls = set()
    unique = []
    for item in all_links:
        if item["url"] not in seen_urls:
            seen_urls.add(item["url"])
            unique.append(item)

    unique.sort(key=lambda x: x["date"])
    log.info("Total unique minutes links found: %d", len(unique))
    return unique


# ── Downloading ───────────────────────────────────────────────────────────────

def download_all(links: list) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for i, item in enumerate(links, 1):
        date_str = item["date"]
        url      = item["url"]
        filename = safe_filename(date_str)
        filepath = OUTPUT_DIR / filename

        log.info("[%d/%d] %s  ->  %s", i, len(links), date_str, filename)

        if filepath.exists():
            log.info("  Already exists, skipping.")
            rows.append({**item, "local_file": filename, "status": "skipped"})
            continue

        try:
            raw_text = fetch_and_extract(url)
            clean    = clean_text(raw_text)

            if len(clean) < 100:
                raise ValueError(
                    f"Extracted text too short ({len(clean)} chars) — likely extraction failed"
                )

            filepath.write_text(clean, encoding="utf-8")
            rows.append({**item, "local_file": filename, "status": "ok"})
            log.info("  Saved (%d chars)", len(clean))

        except Exception as exc:
            log.error("  FAILED: %s", exc)
            rows.append({**item, "local_file": "", "status": f"error: {exc}"})

    # Write index CSV
    with INDEX_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "url", "local_file", "status"])
        writer.writeheader()
        writer.writerows(rows)

    ok    = sum(1 for r in rows if r["status"] == "ok")
    skip  = sum(1 for r in rows if r["status"] == "skipped")
    error = sum(1 for r in rows if r["status"].startswith("error"))
    log.info("Done. Downloaded: %d  |  Skipped: %d  |  Errors: %d", ok, skip, error)
    log.info("Index written to: %s", INDEX_FILE)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    check_dependencies()
    links = discover_all_links()
    download_all(links)