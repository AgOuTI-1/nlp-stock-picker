from __future__ import annotations

import json, os, re, time, socket, urllib.parse
from dataclasses import dataclass
from typing import List, Dict, Any

import feedparser


def _safe_key(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)


@dataclass
class GoogleNewsRSS:
    cache_dir: str = "nlp_cache"
    sleep_s: float = 0.2
    timeout_s: int = 12

    def __post_init__(self):
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, _safe_key(key) + ".json")

    def fetch(self, query: str, max_items: int = 30) -> List[Dict[str, Any]]:
        encoded = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"

        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(self.timeout_s)
        try:
            feed = feedparser.parse(url)
        finally:
            socket.setdefaulttimeout(old_timeout)

        items = []
        for entry in feed.entries[:max_items]:
            items.append({
                "title": getattr(entry, "title", ""),
                "published": getattr(entry, "published", ""),
                "link": getattr(entry, "link", ""),
            })
        time.sleep(self.sleep_s)
        return items

    def get_cached(self, key: str, fetch_fn):
        path = self._cache_path(key)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        data = fetch_fn()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return data
