import sys
if sys.platform.startswith("win"):
    import asyncio
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

from typing import List, Dict
import pandas as pd
from datetime import datetime

from app.collectors.base import Collector
from app.collectors.common import now_iso

# ---------- Playwright helpers ----------
def _p_collect(term: str, limit: int, headless: bool, scroll_batches: int, locale: str, user_agent: str) -> List[Dict]:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

    PIN_URL = "https://www.pinterest.com/search/pins/?q={query}&rs=typed"

    def _dismiss_overlays(page):
        for selector in [
            "button:has-text('Accept')",
            "button:has-text('Accept all')",
            "button[aria-label='Close']",
            "[data-test-id='fullpage-modal-close']",
            "button[title='Close']",
        ]:
            try:
                page.locator(selector).first.click(timeout=800)
                page.wait_for_timeout(200)
            except Exception:
                pass

    def _extract_from_dom(page, limit: int) -> List[Dict]:
        rows: List[Dict] = []
        pins = page.locator("a[href^='/pin/']")
        seen = set()
        n = pins.count()
        for i in range(min(n, max(limit * 2, 60))):
            try:
                el = pins.nth(i)
                href = el.get_attribute("href") or ""
                if not href or href in seen:
                    continue
                seen.add(href)
                alt = el.get_attribute("aria-label") or (el.inner_text(timeout=800) or "")
                full_url = f"https://www.pinterest.com{href}" if href.startswith("/") else href
                rows.append({
                    "source": "Pinterest",
                    "fetched_at": now_iso(),
                    "term": None,
                    "url": full_url,
                    "title": (alt or "").strip()[:200] or None,
                    "text": (alt or "").strip(),
                    "author": None,
                    "region": None,
                    "category": None,
                })
                if len(rows) >= limit:
                    break
            except Exception:
                continue
        return rows

    rows: List[Dict] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent=user_agent,
            locale=locale,
            viewport={"width": 1366, "height": 900},
            extra_http_headers={"Accept-Language": locale},
        )
        page = context.new_page()
        page.set_default_timeout(30000)
        q = term.replace(" ", "%20")

        try:
            page.goto(PIN_URL.format(query=q), wait_until="networkidle")
        except PWTimeoutError:
            page.goto(PIN_URL.format(query=q))

        _dismiss_overlays(page)
        try:
            page.wait_for_selector("a[href^='/pin/']", timeout=8000)
        except PWTimeoutError:
            pass

        # scroll to load content
        for _ in range(scroll_batches):
            page.mouse.wheel(0, 5000)
            page.wait_for_timeout(1400)

        rows.extend(_extract_from_dom(page, limit))

        context.close()
        browser.close()

    # dedupe by URL
    uniq, seen = [], set()
    for r in rows:
        u = r.get("url")
        if u and u not in seen:
            seen.add(u)
            uniq.append(r)
    return uniq[:limit]

# ---------- Selenium fallback ----------
def _s_collect(term: str, limit: int, headless: bool) -> List[Dict]:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from bs4 import BeautifulSoup
    import time

    PIN_URL = "https://www.pinterest.com/search/pins/?q={query}&rs=typed"

    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1366,900")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    try:
        q = term.replace(" ", "%20")
        driver.get(PIN_URL.format(query=q))
        time.sleep(2.0)

        # attempt to close overlays
        for css in ["button[aria-label='Close']"]:
            try:
                driver.find_element("css selector", css).click()
                time.sleep(0.2)
            except Exception:
                pass

        # scroll to load
        for _ in range(5):
            driver.execute_script("window.scrollBy(0, 5000);")
            time.sleep(1.4)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        rows: List[Dict] = []
        seen = set()
        for a in soup.select("a[href^='/pin/']"):
            href = a.get("href")
            if not href or href in seen:
                continue
            seen.add(href)
            full_url = f"https://www.pinterest.com{href}" if href.startswith("/") else href
            alt = a.get("aria-label") or a.get_text(strip=True) or ""
            rows.append({
                "source": "Pinterest",
                "fetched_at": now_iso(),
                "term": None,
                "url": full_url,
                "title": alt[:200] or None,
                "text": alt,
                "author": None,
                "region": None,
                "category": None,
            })
            if len(rows) >= limit:
                break
        return rows
    finally:
        driver.quit()

class PinterestCollector(Collector):
    name = "Pinterest"

    def __init__(self, scroll_batches: int = 5, headless: bool = True, locale: str = "en-US"):
        self.scroll_batches = scroll_batches
        self.headless = headless
        self.locale = locale
        self.ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    def collect(self, term: str, limit: int = 40) -> pd.DataFrame:
        rows: List[Dict] = []
        # 1) Try Playwright
        try:
            rows = _p_collect(term, limit, self.headless, self.scroll_batches, self.locale, self.ua)
        except Exception as e:
            # 2) Fallback to Selenium if Playwright fails (like your NotImplementedError)
            rows = _s_collect(term, limit, self.headless)

        # set term and dedupe
        seen, final = set(), []
        for r in rows:
            r["term"] = term
            u = r.get("url")
            if u and u not in seen:
                seen.add(u)
                final.append(r)
        return pd.DataFrame(final[:limit])
