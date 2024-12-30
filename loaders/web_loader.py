from bs4 import BeautifulSoup
import aiohttp
import asyncio
import time
import itertools
from urllib.parse import urljoin, urlparse
from typing import List, Set, Tuple, Optional
from langchain_core.documents import Document
from .base_loader import DocumentLoader
import logging
from collections import deque
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class URLProcessor:
    def normalize_url(self, base_url: str, url: str) -> str:
        """Normalize URL to handle relative paths and duplicates"""
        return urljoin(base_url, url)
    
    def should_process_url(self, url: str, base_domain: str) -> bool:
        """
        Validate URL belongs to same domain and isn't a binary/unwanted file
        
        Args:
            url: URL to validate
            base_domain: Domain to restrict crawling to
            
        Returns:
            Boolean indicating if URL should be processed
        """
        parsed = urlparse(url)
        
        # Skip empty or invalid URLs
        if not parsed.netloc:
            return False
            
        # Strict domain matching
        if parsed.netloc != base_domain:
            return False
            
        excluded_patterns = {
            # Binary and media files
            '.pdf', '.jpg', '.png', '.gif', '.css', '.js', '.ico',
            # Documentation specific patterns
            '#', 'genindex', 'search', '_sources', '_static',
            'glossary', '_downloads', '_images',
            # Common external links
            'github.com', 'twitter.com', 'linkedin.com',
            # Navigation elements
            'next', 'previous', 'index.html'
        }
        
        return not any(pattern in url.lower() for pattern in excluded_patterns)

class ContentExtractor:
    def extract_content(self, html: str) -> Tuple[str, str]:
        """Extract clean content and title from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove noise elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
            
        # Extract title
        title = soup.title.string if soup.title else ''
        
        # Extract main content with priority for semantic elements
        main_content = (
            soup.find('main') or 
            soup.find('article') or 
            soup.find('div', class_='document') or 
            soup.find('div', class_='content')
        )
        content = main_content.get_text(separator='\n', strip=True) if main_content else soup.get_text(separator='\n', strip=True)
        
        return content, title

class WebsiteLoader(DocumentLoader):
    def __init__(self, chunk_size: Optional[int] = None):
        super().__init__(chunk_size)
        self.url_processor = URLProcessor()
        self.content_extractor = ContentExtractor()
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _process_url(
        self, 
        session: aiohttp.ClientSession,
        url: str,
        visited: Set[str],
        base_domain: str
    ) -> Tuple[Set[str], Optional[Document]]:
        """Process a single URL and return new URLs and document"""
        if url in visited:
            return set(), None
            
        visited.add(url)
        try:
            async with session.get(url) as response:
                if response.status == 429:  # Too Many Requests
                    retry_after = int(response.headers.get('Retry-After', 30))
                    logger.warning(f"Rate limited on {url}, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    return set(), None
                    
                if response.status != 200:
                    logger.warning(f"Status {response.status} for {url}")
                    return set(), None
                    
                html = await response.text()
                content, title = self.content_extractor.extract_content(html)
                
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': url,
                        'title': title,
                        'section': 'web_content'
                    }
                )
                
                soup = BeautifulSoup(html, 'html.parser')
                new_urls = {
                    self.url_processor.normalize_url(url, link['href'])
                    for link in soup.find_all('a', href=True)
                    if self.url_processor.should_process_url(
                        self.url_processor.normalize_url(url, link['href']),
                        base_domain
                    )
                }
                
                return new_urls, doc
                
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            return set(), None

    async def _process_url_batch(
        self,
        session: aiohttp.ClientSession,
        urls: List[Tuple[int, str]],
        visited: Set[str],
        base_domain: str
    ):
        """Process a batch of URLs concurrently"""
        tasks = [
            self._process_url(session, url, visited, base_domain)
            for _, url in urls
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def crawl_website(self, url: str, max_depth: int = 2) -> List[Document]:
        """Crawl website asynchronously with depth control and progress tracking"""
        start_time = time.time()
        processed_count = 0
        base_domain = urlparse(url).netloc
        visited: Set[str] = set()
        documents: List[Document] = []
        
        connector = aiohttp.TCPConnector(
            limit=10,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(
            total=30,
            connect=10,
            sock_read=20
        )
        
        urls_to_process = deque([(0, url)])  # (depth, url)
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'Custom Documentation Crawler 1.0 (Respectful Crawler)',
                'Accept': 'text/html,application/xhtml+xml',
                'Accept-Language': 'en-US,en;q=0.5'
            }
        ) as session:
            while urls_to_process:
                # Process URLs in batches
                current_batch = []
                while urls_to_process and len(current_batch) < 5:
                    url_info = urls_to_process.popleft()
                    if url_info[0] <= max_depth:
                        current_batch.append(url_info)
                
                if not current_batch:
                    continue
                
                results = await self._process_url_batch(session, current_batch, visited, base_domain)
                
                for result, (depth, current_url) in zip(results, current_batch):
                    if isinstance(result, Exception):
                        logger.error(f"Failed processing {current_url}: {str(result)}")
                        continue
                        
                    new_urls, doc = result
                    if doc:
                        documents.append(doc)
                    
                    # Add new URLs with incremented depth
                    for new_url in new_urls:
                        if new_url not in visited:
                            urls_to_process.append((depth + 1, new_url))
                
                processed_count += len(current_batch)
                elapsed = time.time() - start_time
                logger.info(f"Processed {processed_count} URLs in {elapsed:.2f}s")
                
                # Small delay between batches
                await asyncio.sleep(0.1)
        
        total_time = time.time() - start_time
        logger.info(f"Crawl completed: {processed_count} URLs processed in {total_time:.2f}s")
        return documents

    def process_website(self, url: str, max_depth: int = 2) -> List[Document]:
        """Public synchronous interface for website processing"""
        return asyncio.run(self.crawl_website(url, max_depth))

    def reset_processed_files(self) -> None:
        """Implement required method from base class"""
        pass
