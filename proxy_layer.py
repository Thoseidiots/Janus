# proxy_layer.py
# Anonymizing Proxy Layer for Project Manus. This file is immutable.

import requests
from bs4 import BeautifulSoup

TOR_PROXY = {'http': 'socks5h://127.0.0.1:9050', 'https': 'socks5h://127.0.0.1:9050'}
REQUEST_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64 ) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

def fetch_url_via_tor(url: str) -> (str, str):
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, proxies=TOR_PROXY, timeout=20)
        response.raise_for_status()
        return "SUCCESS", response.content
    except requests.exceptions.RequestException as e:
        return "ERROR", f"Tor request failed: {e}"

def parse_html_content(html_content: bytes) -> str:
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        for script_or_style in soup(["script", "style"]): script_or_style.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return '\n'.join(chunk for chunk in chunks if chunk)
    except Exception as e:
        return f"Error parsing HTML content: {e}"

def get_content_anonymously(url: str) -> str:
    status, content = fetch_url_via_tor(url)
    if status == "ERROR": return content
    return parse_html_content(content)
