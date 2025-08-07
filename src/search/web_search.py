"""
Web search fallback module for ElevateAI.
Handles external web search when local content is insufficient.
"""
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse, urljoin
import time

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    from googlesearch import search as google_search
    GOOGLE_SEARCH_AVAILABLE = True
except ImportError:
    GOOGLE_SEARCH_AVAILABLE = False

from config.settings import settings
from src.utils.logger import logger
from src.utils.exceptions import WebSearchError
from src.core.document_processor import DocumentProcessor
from src.analysis.text_cleaner import TextCleaner
from src.analysis.text_chunker import TextChunker
from src.database.embedding_generator import EmbeddingGenerator


@dataclass
class WebSearchResult:
    """Represents a web search result."""
    url: str
    title: str
    snippet: str
    content: str
    relevance_score: float
    source: str


class WebSearchEngine:
    """Web search engine for fallback content retrieval."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the web search engine.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.settings = settings
        self.logger = logger
        
        # Search parameters
        self.max_results = self.config.get('max_results', settings.max_web_search_results)
        self.timeout = self.config.get('timeout', 30)
        self.delay_between_requests = self.config.get('delay', 1.0)
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.text_cleaner = TextCleaner()
        self.text_chunker = TextChunker()
        self.embedding_generator = EmbeddingGenerator()
        
        # Request session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def web_search(self, query: str, **kwargs) -> List[str]:
        """
        Search the web for relevant URLs.
        
        Args:
            query: Search query
            **kwargs: Additional search parameters
            
        Returns:
            List of relevant URLs
            
        Raises:
            WebSearchError: If web search fails
        """
        try:
            self.logger.info(f"Performing web search for: '{query}'")
            
            urls = []
            
            # Try Google Search API first
            if settings.google_api_key and settings.google_cse_id:
                urls.extend(self._google_api_search(query, **kwargs))
            
            # Fallback to googlesearch library
            elif GOOGLE_SEARCH_AVAILABLE:
                urls.extend(self._google_library_search(query, **kwargs))
            
            # Fallback to SerpAPI if available
            elif settings.serpapi_api_key:
                urls.extend(self._serpapi_search(query, **kwargs))
            
            else:
                raise WebSearchError("No web search API configured")
            
            # Remove duplicates and limit results
            unique_urls = list(dict.fromkeys(urls))  # Preserve order
            limited_urls = unique_urls[:self.max_results]
            
            self.logger.info(f"Found {len(limited_urls)} unique URLs")
            return limited_urls
            
        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            raise WebSearchError(f"Web search failed: {e}")
    
    def _google_api_search(self, query: str, **kwargs) -> List[str]:
        """
        Search using Google Custom Search API.
        
        Args:
            query: Search query
            **kwargs: Additional parameters
            
        Returns:
            List of URLs
        """
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': settings.google_api_key,
                'cx': settings.google_cse_id,
                'q': query,
                'num': min(self.max_results, 10)  # Google API limit
            }
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            urls = []
            
            for item in data.get('items', []):
                urls.append(item['link'])
            
            return urls
            
        except Exception as e:
            self.logger.warning(f"Google API search failed: {e}")
            return []
    
    def _google_library_search(self, query: str, **kwargs) -> List[str]:
        """
        Search using googlesearch library.
        
        Args:
            query: Search query
            **kwargs: Additional parameters
            
        Returns:
            List of URLs
        """
        try:
            urls = []
            for url in google_search(query, num_results=self.max_results, sleep_interval=self.delay_between_requests):
                urls.append(url)
                if len(urls) >= self.max_results:
                    break
            
            return urls
            
        except Exception as e:
            self.logger.warning(f"Google library search failed: {e}")
            return []
    
    def _serpapi_search(self, query: str, **kwargs) -> List[str]:
        """
        Search using SerpAPI.
        
        Args:
            query: Search query
            **kwargs: Additional parameters
            
        Returns:
            List of URLs
        """
        try:
            url = "https://serpapi.com/search"
            params = {
                'api_key': settings.serpapi_api_key,
                'engine': 'google',
                'q': query,
                'num': self.max_results
            }
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            urls = []
            
            for result in data.get('organic_results', []):
                urls.append(result['link'])
            
            return urls
            
        except Exception as e:
            self.logger.warning(f"SerpAPI search failed: {e}")
            return []
    
    def scrape_web_content(self, urls: List[str]) -> List[str]:
        """
        Scrape content from web URLs.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of scraped text content
        """
        if not BS4_AVAILABLE:
            raise WebSearchError("BeautifulSoup not available for web scraping")
        
        scraped_content = []
        
        for i, url in enumerate(urls):
            try:
                self.logger.info(f"Scraping content from: {url}")
                
                # Add delay between requests
                if i > 0:
                    time.sleep(self.delay_between_requests)
                
                content = self._scrape_single_url(url)
                if content:
                    scraped_content.append(content)
                
            except Exception as e:
                self.logger.warning(f"Failed to scrape {url}: {e}")
                continue
        
        self.logger.info(f"Successfully scraped {len(scraped_content)} pages")
        return scraped_content
    
    def _scrape_single_url(self, url: str) -> Optional[str]:
        """
        Scrape content from a single URL.
        
        Args:
            url: URL to scrape
            
        Returns:
            Scraped text content or None
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract text from main content areas
            content_selectors = [
                'article', 'main', '.content', '#content', '.post', '.entry'
            ]
            
            content_text = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content_text = elements[0].get_text()
                    break
            
            # Fallback to body text
            if not content_text:
                content_text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in content_text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit content length
            max_length = 10000  # 10k characters
            if len(clean_text) > max_length:
                clean_text = clean_text[:max_length] + "..."
            
            return clean_text
            
        except Exception as e:
            self.logger.warning(f"Failed to scrape {url}: {e}")
            return None
    
    def semantic_search_web(self, query: str, web_contents: List[str]) -> List[Dict[str, Any]]:
        """
        Perform semantic search on web content.
        
        Args:
            query: Search query
            web_contents: List of web content texts
            
        Returns:
            List of relevant chunks with scores
        """
        try:
            if not web_contents:
                return []
            
            self.logger.info(f"Performing semantic search on {len(web_contents)} web documents")
            
            # Process and chunk web content
            all_chunks = []
            for i, content in enumerate(web_contents):
                # Clean content
                cleaned_content = self.text_cleaner.clean_text(content)
                
                # Chunk content
                chunks = self.text_chunker.split_into_chunks(
                    cleaned_content,
                    strategy='semantic'
                )
                
                # Add source information
                for chunk in chunks:
                    chunk.metadata['source'] = f'web_document_{i}'
                    chunk.metadata['content_type'] = 'web_content'
                    all_chunks.append(chunk)
            
            if not all_chunks:
                return []
            
            # Generate embeddings for chunks
            chunk_texts = [chunk.content for chunk in all_chunks]
            embeddings = self.embedding_generator.generate_embeddings_batch(chunk_texts)
            
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # Calculate similarities
            results = []
            for chunk, embedding in zip(all_chunks, embeddings):
                similarity = self.embedding_generator.calculate_similarity(
                    query_embedding, embedding
                )
                
                if similarity >= settings.similarity_threshold * 0.8:  # Lower threshold for web content
                    result = {
                        'text': chunk.content,
                        'score': similarity,
                        'metadata': chunk.metadata,
                        'chunk_id': chunk.chunk_id
                    }
                    results.append(result)
            
            # Sort by similarity score
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # Limit results
            top_results = results[:settings.rerank_top_k]
            
            self.logger.info(f"Found {len(top_results)} relevant web chunks")
            return top_results
            
        except Exception as e:
            self.logger.error(f"Web semantic search failed: {e}")
            raise WebSearchError(f"Web semantic search failed: {e}")
    
    def fallback_search(self, query: str, local_results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Perform fallback web search when local results are insufficient.
        
        Args:
            query: Search query
            local_results: Local search results
            
        Returns:
            Tuple of (combined_results, used_web_search)
        """
        try:
            # Check if local results are sufficient
            if self._are_local_results_sufficient(local_results):
                return local_results, False
            
            self.logger.info("Local results insufficient, performing web search fallback")
            
            # Perform web search
            urls = self.web_search(query)
            if not urls:
                return local_results, False
            
            # Scrape web content
            web_contents = self.scrape_web_content(urls)
            if not web_contents:
                return local_results, False
            
            # Search web content
            web_results = self.semantic_search_web(query, web_contents)
            
            # Mark web results
            for result in web_results:
                result['metadata']['is_web_result'] = True
                result['metadata']['search_query'] = query
            
            # Combine local and web results
            combined_results = local_results + web_results
            
            # Sort by score
            combined_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Limit total results
            final_results = combined_results[:settings.max_chunks_per_query]
            
            self.logger.info(f"Combined {len(local_results)} local + {len(web_results)} web results")
            return final_results, True
            
        except Exception as e:
            self.logger.error(f"Fallback search failed: {e}")
            return local_results, False
    
    def _are_local_results_sufficient(self, results: List[Dict[str, Any]]) -> bool:
        """
        Check if local results are sufficient.
        
        Args:
            results: Local search results
            
        Returns:
            True if results are sufficient
        """
        if not results:
            return False
        
        # Check number of high-quality results
        high_quality_results = [
            result for result in results
            if result.get('score', 0) >= settings.similarity_threshold
        ]
        
        # Need at least 3 high-quality results
        return len(high_quality_results) >= 3
    
    def get_web_search_stats(self) -> Dict[str, Any]:
        """
        Get web search statistics.
        
        Returns:
            Dictionary with web search statistics
        """
        return {
            'max_results': self.max_results,
            'timeout': self.timeout,
            'delay_between_requests': self.delay_between_requests,
            'google_api_available': bool(settings.google_api_key and settings.google_cse_id),
            'serpapi_available': bool(settings.serpapi_api_key),
            'google_library_available': GOOGLE_SEARCH_AVAILABLE,
            'bs4_available': BS4_AVAILABLE
        }
