import pandas as pd
import requests
from app.collectors.base import Collector
from app.collectors.common import now_iso, HEADERS



class RedditCollector(Collector):
    name = "Reddit"

    def __init__(self, subreddit: str | None = None):
        self.subreddit = subreddit  # if None, search all

    def collect(self, term: str, limit: int = 50) -> pd.DataFrame:
        # Use Reddit JSON search endpoint (public). For heavier use, prefer official API.
        if self.subreddit:
            url = f"https://www.reddit.com/r/{self.subreddit}/search.json?q={term}&restrict_sr=1&sort=new"
        else:
            url = f"https://www.reddit.com/search.json?q={term}&sort=new"
        r = requests.get(url, headers={**HEADERS, "User-Agent": "Mozilla/5.0 RedditCollector"}, timeout=20)
        r.raise_for_status()
        data = r.json()
        rows = []
        for child in data.get("data", {}).get("children", [])[:limit]:
            d = child.get("data", {})
            rows.append({
                "source": self.name,
                "fetched_at": now_iso(),
                "term": term,
                "url": f"https://www.reddit.com{d.get('permalink')}",
                "title": d.get("title"),
                "text": d.get("selftext") or d.get("title"),
                "author": d.get("author"),
                "region": None,
                "category": d.get("subreddit"),
            })
        return pd.DataFrame(rows)