import pandas as pd
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from app.collectors.base import Collector
from app.collectors.common import now_iso, HEADERS

class YahooNewsCollector(Collector):
    name = "Yahoo News"

    def __init__(self, use_selenium: bool = False):
        self.use_selenium = use_selenium

    def _requests_collect(self, term: str, limit: int = 30) -> pd.DataFrame:
        q = term.replace(" ", "+")
        url = f"https://news.search.yahoo.com/search?p={q}"
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        cards = soup.select("div.NewsArticle")
        rows = []
        for c in cards[:limit]:
            a = c.select_one("h4 a")
            title = a.get_text(strip=True) if a else None
            link = a.get("href") if a else None
            snippet = c.select_one("p.s-textLine2").get_text(strip=True) if c.select_one("p.s-textLine2") else None
            rows.append({
                "source": self.name,
                "fetched_at": now_iso(),
                "term": term,
                "url": link,
                "title": title,
                "text": snippet,
                "author": None,
                "region": None,
                "category": None,
            })
        return pd.DataFrame(rows)

    def _selenium_collect(self, term: str, limit: int = 30) -> pd.DataFrame:
        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
        try:
            q = term.replace(" ", "+")
            driver.get(f"https://news.search.yahoo.com/search?p={q}")
            soup = BeautifulSoup(driver.page_source, "html.parser")
            cards = soup.select("div.NewsArticle")
            rows = []
            for c in cards[:limit]:
                a = c.select_one("h4 a")
                title = a.get_text(strip=True) if a else None
                link = a.get("href") if a else None
                snippet = c.select_one("p.s-textLine2").get_text(strip=True) if c.select_one("p.s-textLine2") else None
                rows.append({
                    "source": self.name,
                    "fetched_at": now_iso(),
                    "term": term,
                    "url": link,
                    "title": title,
                    "text": snippet,
                    "author": None,
                    "region": None,
                    "category": None,
                })
            return pd.DataFrame(rows)
        finally:
            driver.quit()

    def collect(self, term: str, limit: int = 30) -> pd.DataFrame:
        if self.use_selenium:
            return self._selenium_collect(term, limit)
        return self._requests_collect(term, limit)