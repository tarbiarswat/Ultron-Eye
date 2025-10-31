import pandas as pd
import feedparser
from app.collectors.base import Collector
from app.collectors.common import now_iso

class RSSCollector(Collector):
    name = "Google RSS"

    def __init__(self, feed_template: str = "https://news.google.com/rss/search?q={query}"):
        self.feed_template = feed_template

    def collect(self, term: str, limit: int = 50) -> pd.DataFrame:
        url = self.feed_template.format(query=term.replace(" ", "+"))
        feed = feedparser.parse(url)
        rows = []
        for entry in feed.entries[:limit]:
            rows.append({
                "source": self.name,
                "fetched_at": now_iso(),
                "term": term,
                "url": entry.get("link"),
                "title": entry.get("title"),
                "text": entry.get("summary", ""),
                "author": entry.get("author"),
                "region": None,
                "category": None,
            })
        return pd.DataFrame(rows)