import time
import random
import requests
from bs4 import BeautifulSoup
from datetime import datetime

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}

def now_iso():
    return datetime.utcnow().isoformat()

def soup_from_url(url: str, sleep=(0.5, 1.2)) -> BeautifulSoup:
    time.sleep(random.uniform(*sleep))
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")