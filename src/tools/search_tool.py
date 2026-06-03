"""
Search Tool - Multi-source search with multiple fallback strategies.
Sources: ddgs library → DDG HTML scrape → Wikipedia REST API → Wikipedia-api library
"""
import asyncio
import re
import urllib.parse
from typing import Optional
from loguru import logger

try:
    from ddgs import DDGS
    DDG_AVAILABLE = True
    DDG_NEW = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDG_AVAILABLE = True
        DDG_NEW = False
    except ImportError:
        DDG_AVAILABLE = False
        DDG_NEW = False

try:
    import wikipediaapi
    WIKI_LIB_AVAILABLE = True
except ImportError:
    WIKI_LIB_AVAILABLE = False

try:
    import httpx
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False

from src.models.schemas import SearchResult


class SearchTool:
    """Multi-source search with cascading fallbacks."""

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    def __init__(self):
        self.wiki = None
        if WIKI_LIB_AVAILABLE:
            try:
                self.wiki = wikipediaapi.Wikipedia(
                    language="en",
                    extract_format=wikipediaapi.ExtractFormat.WIKI,
                    user_agent="AgenticFactChecker/1.0"
                )
            except Exception as e:
                logger.warning(f"Wikipedia-api init failed: {e}")

        logger.info(
            f"SearchTool initialized | DDG={DDG_AVAILABLE} (new={DDG_NEW}) | "
            f"WikiLib={WIKI_LIB_AVAILABLE} | Scraping={SCRAPING_AVAILABLE}"
        )

    # ── Main entry point ────────────────────────────────────────

    async def search_all(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search all sources with fallbacks."""
        results = []

        # Run DDG + Wikipedia in parallel
        ddg_task = asyncio.create_task(self._search_ddg_all_methods(query, max_results - 1))
        wiki_task = asyncio.create_task(self._search_wikipedia_all_methods(query))

        ddg_results, wiki_results = await asyncio.gather(ddg_task, wiki_task, return_exceptions=True)

        if isinstance(ddg_results, list):
            results.extend(ddg_results)
        if isinstance(wiki_results, list):
            results.extend(wiki_results)

        # Deduplicate by URL
        seen, unique = set(), []
        for r in results:
            if r.url not in seen:
                seen.add(r.url)
                unique.append(r)

        logger.info(f"Total unique results for '{query[:50]}': {len(unique)}")
        return unique[:max_results]

    # ── DDG: 3 methods ──────────────────────────────────────────

    async def _search_ddg_all_methods(self, query: str, max_results: int) -> list[SearchResult]:
        """Try ddgs library → DDG HTML → DDG lite."""

        results = await self._ddg_library(query, max_results)
        if results:
            return results

        logger.info("DDG library returned 0, trying HTML scrape...")
        results = await self._ddg_html_scrape(query, max_results)
        if results:
            return results

        logger.info("DDG HTML returned 0, trying lite...")
        results = await self._ddg_lite_scrape(query, max_results)
        return results

    async def _ddg_library(self, query: str, max_results: int) -> list[SearchResult]:
        if not DDG_AVAILABLE:
            return []
        try:
            loop = asyncio.get_event_loop()

            def _search():
                with DDGS(timeout=10) as ddgs:
                    return list(ddgs.text(query, max_results=max_results))

            raw = await asyncio.wait_for(loop.run_in_executor(None, _search), timeout=15)
            results = [
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("href", "") or item.get("url", ""),
                    snippet=item.get("body", "") [:500],
                    source_type="web"
                )
                for item in raw if item.get("href") or item.get("url")
            ]
            logger.info(f"DDG library: {len(results)} results")
            return results
        except Exception as e:
            logger.warning(f"DDG library failed: {e}")
            return []

    async def _ddg_html_scrape(self, query: str, max_results: int) -> list[SearchResult]:
        if not SCRAPING_AVAILABLE:
            return []
        try:
            encoded = urllib.parse.quote_plus(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded}"
            async with httpx.AsyncClient(timeout=12, headers=self.HEADERS, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "lxml")
            results = []
            for result in soup.select(".result__body")[:max_results]:
                title_el = result.select_one(".result__title")
                link_el  = result.select_one(".result__url")
                snip_el  = result.select_one(".result__snippet")
                if not title_el:
                    continue
                title   = title_el.get_text(strip=True)
                raw_url = link_el.get_text(strip=True) if link_el else ""
                if raw_url and not raw_url.startswith("http"):
                    raw_url = "https://" + raw_url
                snippet = snip_el.get_text(strip=True)[:500] if snip_el else ""
                if title and raw_url:
                    results.append(SearchResult(title=title, url=raw_url, snippet=snippet, source_type="web"))

            logger.info(f"DDG HTML scrape: {len(results)} results")
            return results
        except Exception as e:
            logger.warning(f"DDG HTML scrape failed: {e}")
            return []

    async def _ddg_lite_scrape(self, query: str, max_results: int) -> list[SearchResult]:
        if not SCRAPING_AVAILABLE:
            return []
        try:
            encoded = urllib.parse.quote_plus(query)
            url = f"https://lite.duckduckgo.com/lite/?q={encoded}"
            async with httpx.AsyncClient(timeout=12, headers=self.HEADERS, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "lxml")
            results = []
            rows = soup.find_all("tr")
            i = 0
            while i < len(rows) and len(results) < max_results:
                row = rows[i]
                link = row.find("a", {"class": "result-link"}) or row.find("a")
                if link and link.get("href", "").startswith("http"):
                    title   = link.get_text(strip=True)
                    href    = link["href"]
                    snippet = rows[i + 1].get_text(strip=True)[:400] if i + 1 < len(rows) else ""
                    if title:
                        results.append(SearchResult(title=title, url=href, snippet=snippet, source_type="web"))
                i += 1

            logger.info(f"DDG lite: {len(results)} results")
            return results
        except Exception as e:
            logger.warning(f"DDG lite failed: {e}")
            return []

    # ── Wikipedia: 2 methods ────────────────────────────────────

    async def _search_wikipedia_all_methods(self, query: str) -> list[SearchResult]:
        results = await self._wikipedia_rest_api(query)
        if results:
            return results
        return await self._wikipedia_lib(query)

    async def _wikipedia_rest_api(self, query: str) -> list[SearchResult]:
        """Wikipedia public REST API — very reliable, no key needed."""
        if not SCRAPING_AVAILABLE:
            return []
        try:
            encoded = urllib.parse.quote_plus(query)
            search_url = (
                f"https://en.wikipedia.org/w/api.php"
                f"?action=query&list=search&srsearch={encoded}"
                f"&format=json&srlimit=2"
            )
            wiki_headers = {"User-Agent": "AgenticFactChecker/1.0"}
            async with httpx.AsyncClient(timeout=12, headers=wiki_headers) as client:
                resp = await client.get(search_url)
                resp.raise_for_status()
                data = resp.json()

                results = []
                for item in data.get("query", {}).get("search", [])[:2]:
                    title    = item.get("title", "")
                    raw_snip = item.get("snippet", "")
                    snippet  = re.sub(r'<[^>]+>', '', raw_snip)[:500]
                    url = (
                        "https://en.wikipedia.org/wiki/"
                        + urllib.parse.quote(title.replace(" ", "_"))
                    )
                    # Try to fetch a richer summary
                    try:
                        sum_url = (
                            "https://en.wikipedia.org/api/rest_v1/page/summary/"
                            + urllib.parse.quote(title.replace(" ", "_"))
                        )
                        sr = await client.get(sum_url)
                        if sr.status_code == 200:
                            snippet = sr.json().get("extract", snippet)[:600]
                    except Exception:
                        pass

                    if title:
                        results.append(SearchResult(
                            title=title, url=url, snippet=snippet, source_type="wikipedia"
                        ))

            logger.info(f"Wikipedia REST API: {len(results)} results")
            return results
        except Exception as e:
            logger.warning(f"Wikipedia REST API failed: {e}")
            return []

    async def _wikipedia_lib(self, query: str) -> list[SearchResult]:
        if not self.wiki:
            return []
        try:
            loop = asyncio.get_event_loop()

            def _fetch():
                for q in [query, " ".join(query.split()[:3])]:
                    page = self.wiki.page(q)
                    if page.exists():
                        return [SearchResult(
                            title=page.title,
                            url=page.fullurl,
                            snippet=page.summary[:600],
                            source_type="wikipedia"
                        )]
                return []

            return await asyncio.wait_for(loop.run_in_executor(None, _fetch), timeout=12)
        except Exception as e:
            logger.warning(f"Wikipedia lib fallback failed: {e}")
            return []

    # ── Content enrichment ──────────────────────────────────────

    async def get_enriched_content(self, results: list[SearchResult]) -> list[str]:
        tasks   = [self._scrape_page(r.url) for r in results]
        scraped = await asyncio.gather(*tasks, return_exceptions=True)
        enriched = []
        for result, content in zip(results, scraped):
            if isinstance(content, str) and len(content) > 100:
                enriched.append(f"[{result.title}] {content}")
            else:
                enriched.append(f"[{result.title}] {result.snippet}")
        return enriched

    async def _scrape_page(self, url: str, max_chars: int = 2000) -> Optional[str]:
        if not SCRAPING_AVAILABLE:
            return None
        skip = ["youtube.com", "twitter.com", "facebook.com", "instagram.com", "tiktok.com"]
        if any(d in url for d in skip):
            return None
        try:
            async with httpx.AsyncClient(timeout=8, headers=self.HEADERS, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "lxml")
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                main = soup.find("main") or soup.find("article") or soup.find("body")
                if main:
                    return " ".join(main.get_text(separator=" ").split())[:max_chars]
        except Exception as e:
            logger.debug(f"Scrape failed for {url}: {e}")
        return None

    # ── Direct access for /search/ endpoint ─────────────────────

    async def search_duckduckgo(self, query: str, max_results: int = 5) -> list[SearchResult]:
        return await self._search_ddg_all_methods(query, max_results)

    async def search_wikipedia(self, query: str, max_results: int = 2) -> list[SearchResult]:
        return await self._search_wikipedia_all_methods(query)