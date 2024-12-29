from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
from typing import List, Set, Optional
from langchain_community.document_loaders import WebBaseLoader
from .base_loader import DocumentLoader
import logging

logger = logging.getLogger(__name__)

class WebsiteLoader(DocumentLoader):
    def __init__(self, max_depth: int = 2):
        super().__init__()
        self.max_depth = max_depth
        self.visited_urls = set()
    
    def process_website(self, url: str, processed_docs: dict, custom_metadata: Optional[dict] = None) -> List:
        """
        Main entry point for website processing:
        1. Crawls website up to max_depth
        2. Extracts content from each page
        3. Handles deduplication via processed_docs
        4. Adds metadata to documents
        """
        documents = []
        try:
            docs = self._process_url(url)
            for doc in docs:
                doc_hash = self._get_hash(doc.page_content)
                if doc_hash not in processed_docs:
                    doc.metadata.update({
                        "source": url,
                        "content_type": "website",
                        "url": url,
                        **(custom_metadata or {})
                    })
                    documents.append(doc)
                    processed_docs[doc_hash] = url  # Track processed content
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
        return documents
    
    def _get_links(self, url: str, html_content: str) -> Set[str]:
        """Extract all links from the same domain"""
        base_domain = urlparse(url).netloc
        soup = BeautifulSoup(html_content, 'html.parser')
        links = set()
        
        for link in soup.find_all('a', href=True):
            href = urljoin(url, link['href'])
            parsed_href = urlparse(href)
            # Add domain and path validation
            if (parsed_href.netloc == base_domain and 
                not href.endswith(('.pdf', '.zip', '.jpg', '.png'))):
                links.add(href)
        return links
    
    def _process_url(self, url: str, depth: int = 0) -> List:
        """Recursively process URLs up to max_depth"""
        if depth >= self.max_depth or url in self.visited_urls:
            return []
        
        self.visited_urls.add(url)
        documents = []
        
        try:
            # Add request headers and timeout
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Check for HTTP errors
            
            # Load content
            loader = WebBaseLoader(url)
            docs = loader.load()
            documents.extend(docs)
            
            # Process linked pages
            links = self._get_links(url, response.text)
            for link in links:
                documents.extend(self._process_url(link, depth + 1))
                
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            
        return documents
