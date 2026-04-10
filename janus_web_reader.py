"""
janus_web_reader.py
====================
Real web browsing with comprehension for Janus.

Janus can now:
  - Navigate to any URL
  - Extract and understand the page content
  - Answer questions about what it read
  - Follow links intelligently
  - Research topics across multiple pages
  - Extract structured data (prices, tables, lists)

No API keys. Uses urllib (stdlib) for fetching and
html.parser (stdlib) for parsing. JanusBrain for comprehension.

Usage:
    from janus_web_reader import JanusWebReader
    reader = JanusWebReader()

    # Read and understand a page
    content = reader.read("https://news.ycombinator.com")
    answer  = reader.ask("https://docs.python.org/3/library/json.html",
                         "How do I parse a JSON string?")

    # Research a topic
    summary = reader.research("Python asyncio best practices", max_pages=3)
"""

from __future__ import annotations

import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── HTML → clean text parser ──────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    """Strips HTML tags and extracts readable text + links."""

    SKIP_TAGS = {"script", "style", "noscript", "head", "meta", "link",
                 "svg", "path", "iframe", "nav", "footer", "header"}

    def __init__(self):
        super().__init__()
        self._skip    = 0
        self._chunks: List[str] = []
        self.links:   List[Tuple[str, str]] = []  # (href, text)
        self._cur_href = ""
        self._cur_link_text: List[str] = []
        self._in_link = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip += 1
        if tag == "a":
            self._in_link = True
            self._cur_link_text = []
            for k, v in attrs:
                if k == "href" and v:
                    self._cur_href = v
        if tag in ("br", "p", "div", "li", "h1", "h2", "h3", "h4", "tr"):
            self._chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        if tag == "a" and self._in_link:
            link_text = "".join(self._cur_link_text).strip()
            if link_text and self._cur_href:
                self.links.append((self._cur_href, link_text))
            self._in_link = False
            self._cur_href = ""

    def handle_data(self, data):
        if self._skip:
            return
        text = data.strip()
        if text:
            self._chunks.append(text)
            if self._in_link:
                self._cur_link_text.append(text)

    def get_text(self) -> str:
        raw = " ".join(self._chunks)
        # Collapse whitespace
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        raw = re.sub(r" {2,}", " ", raw)
        return html.unescape(raw).strip()


# ── Page result ───────────────────────────────────────────────────────────────

@dataclass
class WebPage:
    url:       str
    title:     str
    text:      str           # clean readable text
    links:     List[Tuple[str, str]]  # (href, anchor_text)
    fetched_at: str          = field(default_factory=lambda: datetime.now().isoformat())
    status:    int           = 200
    error:     Optional[str] = None

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def summary(self, max_chars: int = 500) -> str:
        return self.text[:max_chars] + ("..." if len(self.text) > max_chars else "")

    def extract_numbers(self) -> List[str]:
        """Extract all numbers/prices from the page."""
        return re.findall(r'\$[\d,]+\.?\d*|\d+\.?\d*%|\b\d{4,}\b', self.text)

    def extract_emails(self) -> List[str]:
        return re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', self.text)

    def to_dict(self) -> dict:
        return {
            "url":        self.url,
            "title":      self.title,
            "word_count": self.word_count,
            "text":       self.text[:2000],
            "links":      self.links[:20],
            "fetched_at": self.fetched_at,
        }


# ── Web reader ────────────────────────────────────────────────────────────────

class JanusWebReader:
    """
    Reads and understands web pages.
    Combines fetching, parsing, and brain comprehension.
    """

    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    TIMEOUT    = 15
    MAX_BYTES  = 500_000   # 500KB max per page
    _CACHE: Dict[str, WebPage] = {}

    def fetch(self, url: str, use_cache: bool = True) -> WebPage:
        """Fetch and parse a web page. Returns a WebPage object."""
        if use_cache and url in self._CACHE:
            return self._CACHE[url]

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": self.USER_AGENT,
                         "Accept": "text/html,application/xhtml+xml,*/*",
                         "Accept-Language": "en-US,en;q=0.9"},
            )
            with urllib.request.urlopen(req, timeout=self.TIMEOUT) as resp:
                raw = resp.read(self.MAX_BYTES).decode("utf-8", errors="replace")
                status = resp.status

            # Extract title
            title_m = re.search(r"<title[^>]*>([^<]+)</title>", raw, re.IGNORECASE)
            title   = html.unescape(title_m.group(1).strip()) if title_m else url

            # Parse text and links
            parser = _TextExtractor()
            parser.feed(raw)
            text  = parser.get_text()
            links = self._resolve_links(parser.links, url)

            page = WebPage(url=url, title=title, text=text,
                           links=links, status=status)
            self._CACHE[url] = page
            return page

        except urllib.error.HTTPError as e:
            return WebPage(url=url, title="", text="", links=[],
                           status=e.code, error=str(e))
        except Exception as e:
            return WebPage(url=url, title="", text="", links=[],
                           status=0, error=str(e))

    def read(self, url: str) -> str:
        """Fetch a page and return its clean text."""
        page = self.fetch(url)
        if page.error:
            return f"Could not read {url}: {page.error}"
        return page.text[:8000]  # cap for brain context

    def ask(self, url: str, question: str) -> str:
        """
        Fetch a page and answer a question about it using JanusBrain.
        This is the core comprehension capability.
        """
        page = self.fetch(url)
        if page.error:
            return f"Could not read page: {page.error}"

        # Trim to fit brain context
        content = page.text[:6000]

        try:
            from avus_brain import get_brain
            brain = get_brain()
            prompt = (
                f"Based on this web page content, answer the question.\n\n"
                f"Page: {page.title} ({url})\n\n"
                f"Content:\n{content}\n\n"
                f"Question: {question}"
            )
            return brain.ask(prompt, max_tokens=400)
        except Exception as e:
            # Fallback: keyword search in text
            return self._keyword_answer(page.text, question)

    def research(self, topic: str, max_pages: int = 3) -> str:
        """
        Research a topic by searching and reading multiple pages.
        Returns a synthesized summary.
        """
        # Use DuckDuckGo HTML search (no API key)
        search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(topic)}"
        search_page = self.fetch(search_url)

        # Extract result URLs from search page
        result_urls = []
        for href, text in search_page.links:
            if href.startswith("http") and "duckduckgo" not in href:
                result_urls.append((href, text))
            if len(result_urls) >= max_pages:
                break

        if not result_urls:
            return f"Could not find search results for: {topic}"

        # Read each result
        summaries = []
        for url, link_text in result_urls[:max_pages]:
            try:
                page = self.fetch(url)
                if not page.error and page.word_count > 100:
                    summaries.append(f"Source: {page.title}\n{page.summary(300)}")
                time.sleep(0.5)  # polite delay
            except Exception:
                pass

        if not summaries:
            return f"Could not read any pages about: {topic}"

        combined = "\n\n---\n\n".join(summaries)

        try:
            from avus_brain import get_brain
            brain = get_brain()
            return brain.summarize(
                f"Research topic: {topic}\n\nSources:\n{combined}"
            )
        except Exception:
            return f"Research on '{topic}':\n\n" + combined[:2000]

    def extract_data(self, url: str, data_type: str) -> List[str]:
        """
        Extract specific data types from a page.
        data_type: "prices" | "emails" | "links" | "numbers" | "headings"
        """
        page = self.fetch(url)
        if page.error:
            return []

        if data_type == "prices":
            return re.findall(r'\$[\d,]+\.?\d*', page.text)
        elif data_type == "emails":
            return page.extract_emails()
        elif data_type == "links":
            return [href for href, _ in page.links if href.startswith("http")]
        elif data_type == "numbers":
            return page.extract_numbers()
        elif data_type == "headings":
            return re.findall(r'\n([A-Z][^.\n]{10,80})\n', page.text)
        return []

    def follow_and_read(self, start_url: str, link_keyword: str) -> Optional[WebPage]:
        """
        Read a page, find a link matching a keyword, follow it.
        Useful for navigating sites autonomously.
        """
        page = self.fetch(start_url)
        for href, text in page.links:
            if link_keyword.lower() in text.lower() or link_keyword.lower() in href.lower():
                return self.fetch(href)
        return None

    def _resolve_links(self, links: List[Tuple[str, str]], base_url: str) -> List[Tuple[str, str]]:
        """Convert relative links to absolute."""
        resolved = []
        for href, text in links:
            if href.startswith("http"):
                resolved.append((href, text))
            elif href.startswith("/"):
                parsed = urllib.parse.urlparse(base_url)
                resolved.append((f"{parsed.scheme}://{parsed.netloc}{href}", text))
        return resolved[:50]

    def _keyword_answer(self, text: str, question: str) -> str:
        """Simple keyword-based answer when brain unavailable."""
        words    = set(question.lower().split())
        lines    = text.split("\n")
        relevant = [l for l in lines if any(w in l.lower() for w in words) and len(l) > 20]
        return "\n".join(relevant[:5]) if relevant else "Could not find relevant information."


# ── Module-level singleton ────────────────────────────────────────────────────

_reader: Optional[JanusWebReader] = None

def get_reader() -> JanusWebReader:
    global _reader
    if _reader is None:
        _reader = JanusWebReader()
    return _reader


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--read",     type=str, help="Read a URL")
    parser.add_argument("--ask",      type=str, help="Question to ask about the page")
    parser.add_argument("--research", type=str, help="Research a topic")
    parser.add_argument("--extract",  type=str, help="Data type to extract (prices/emails/links)")
    args = parser.parse_args()

    reader = JanusWebReader()

    if args.read:
        if args.ask:
            print(reader.ask(args.read, args.ask))
        elif args.extract:
            data = reader.extract_data(args.read, args.extract)
            print(f"Found {len(data)} {args.extract}:")
            for item in data[:20]:
                print(f"  {item}")
        else:
            print(reader.read(args.read)[:2000])
    elif args.research:
        print(reader.research(args.research))
    else:
        parser.print_help()
