"""
Multi-hop retrieval engine for ElevateAI.
Handles complex queries with iterative retrieval and context building.
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

from config.settings import settings
from src.utils.logger import logger
from src.utils.exceptions import SearchError
from .semantic_search import SemanticSearchEngine, SearchResult


@dataclass
class RetrievalContext:
    """Context for multi-hop retrieval."""
    original_query: str
    current_query: str
    hop_number: int
    accumulated_results: List[SearchResult]
    context_keywords: List[str]
    confidence_score: float


class RetrievalEngine:
    """Multi-hop retrieval engine for complex queries."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the retrieval engine.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.settings = settings
        self.logger = logger
        
        # Retrieval parameters
        self.max_hops = self.config.get('max_hops', 3)
        self.min_confidence = self.config.get('min_confidence', 0.3)
        self.context_window = self.config.get('context_window', 5)
        self.max_chunks_per_query = self.config.get(
            'max_chunks_per_query', 
            settings.max_chunks_per_query
        )
        
        # Initialize search engine
        self.search_engine = SemanticSearchEngine(config)
    
    def multi_hop_retrieval(self, query: str, **kwargs) -> List[SearchResult]:
        """
        Perform multi-hop retrieval for complex queries.
        
        Args:
            query: Complex search query
            **kwargs: Additional retrieval parameters
            
        Returns:
            List of retrieved results from all hops
            
        Raises:
            SearchError: If retrieval fails
        """
        try:
            self.logger.info(f"Starting multi-hop retrieval for: '{query}'")
            
            # Initialize retrieval context
            context = RetrievalContext(
                original_query=query,
                current_query=query,
                hop_number=0,
                accumulated_results=[],
                context_keywords=self._extract_keywords(query),
                confidence_score=1.0
            )
            
            # Perform iterative retrieval
            for hop in range(self.max_hops):
                context.hop_number = hop + 1
                
                self.logger.info(f"Hop {context.hop_number}: Searching for '{context.current_query}'")
                
                # Perform search for current hop
                hop_results = self._perform_hop_search(context, **kwargs)
                
                if not hop_results:
                    self.logger.info(f"No results found in hop {context.hop_number}, stopping")
                    break
                
                # Add results to accumulated results
                context.accumulated_results.extend(hop_results)
                
                # Check if we have enough high-quality results
                if self._should_stop_retrieval(context):
                    self.logger.info(f"Sufficient results found, stopping at hop {context.hop_number}")
                    break
                
                # Generate next query based on current results
                next_query = self._generate_next_query(context, hop_results)
                if not next_query or next_query == context.current_query:
                    self.logger.info("No new query generated, stopping retrieval")
                    break
                
                context.current_query = next_query
                context.confidence_score *= 0.8  # Decay confidence with each hop
            
            # Deduplicate and rank final results
            final_results = self._deduplicate_and_rank(context.accumulated_results)
            
            self.logger.info(f"Multi-hop retrieval completed: {len(final_results)} unique results")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Multi-hop retrieval failed: {e}")
            raise SearchError(f"Multi-hop retrieval failed: {e}")
    
    def _perform_hop_search(self, context: RetrievalContext, **kwargs) -> List[SearchResult]:
        """
        Perform search for a single hop.
        
        Args:
            context: Current retrieval context
            **kwargs: Search parameters
            
        Returns:
            Results for this hop
        """
        try:
            # Adjust search parameters based on hop number
            k = kwargs.get('k', self.max_chunks_per_query)
            threshold = kwargs.get('threshold', self.settings.similarity_threshold)
            
            # Lower threshold for later hops to explore more
            if context.hop_number > 1:
                threshold *= 0.8
                k = min(k * 2, 20)  # Increase search breadth
            
            # Perform search
            results = self.search_engine.search(
                context.current_query,
                k=k,
                threshold=threshold,
                **kwargs
            )
            
            # Filter out results we already have
            existing_ids = {result.id for result in context.accumulated_results}
            new_results = [result for result in results if result.id not in existing_ids]
            
            return new_results
            
        except Exception as e:
            self.logger.warning(f"Hop search failed: {e}")
            return []
    
    def _should_stop_retrieval(self, context: RetrievalContext) -> bool:
        """
        Determine if retrieval should stop.
        
        Args:
            context: Current retrieval context
            
        Returns:
            True if retrieval should stop
        """
        # Stop if we have enough high-quality results
        high_quality_results = [
            result for result in context.accumulated_results
            if result.score >= self.settings.similarity_threshold
        ]
        
        if len(high_quality_results) >= self.max_chunks_per_query:
            return True
        
        # Stop if confidence is too low
        if context.confidence_score < self.min_confidence:
            return True
        
        return False
    
    def _generate_next_query(self, context: RetrievalContext, hop_results: List[SearchResult]) -> Optional[str]:
        """
        Generate the next query based on current results.
        
        Args:
            context: Current retrieval context
            hop_results: Results from current hop
            
        Returns:
            Next query string or None
        """
        if not hop_results:
            return None
        
        try:
            # Extract key terms from top results
            top_results = hop_results[:3]  # Use top 3 results
            
            # Collect important terms from results
            important_terms = set()
            for result in top_results:
                text = result.text.lower()
                
                # Extract noun phrases and important keywords
                terms = self._extract_important_terms(text)
                important_terms.update(terms)
            
            # Remove terms already in original query
            original_terms = set(context.original_query.lower().split())
            new_terms = important_terms - original_terms
            
            if not new_terms:
                return None
            
            # Create new query by combining original query with new terms
            selected_terms = list(new_terms)[:3]  # Limit to 3 new terms
            next_query = context.original_query + " " + " ".join(selected_terms)
            
            return next_query
            
        except Exception as e:
            self.logger.warning(f"Failed to generate next query: {e}")
            return None
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction (can be improved with NLP)
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords[:10]  # Return top 10 keywords
    
    def _extract_important_terms(self, text: str) -> List[str]:
        """
        Extract important terms from text for query expansion.
        
        Args:
            text: Input text
            
        Returns:
            List of important terms
        """
        # Simple term extraction (can be improved with NLP)
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter for potentially important terms
        important_terms = []
        for word in words:
            # Skip very short or very long words
            if 3 <= len(word) <= 15:
                # Skip common words
                if word not in {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}:
                    important_terms.append(word)
        
        # Remove duplicates and return top terms
        unique_terms = list(set(important_terms))
        return unique_terms[:5]
    
    def _deduplicate_and_rank(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove duplicates and re-rank results.
        
        Args:
            results: List of search results
            
        Returns:
            Deduplicated and ranked results
        """
        # Remove duplicates by ID
        seen_ids = set()
        unique_results = []
        
        for result in results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                unique_results.append(result)
        
        # Sort by score (descending)
        ranked_results = sorted(unique_results, key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(ranked_results):
            result.rank = i + 1
        
        # Limit to max chunks
        return ranked_results[:self.max_chunks_per_query]
    
    def contextual_retrieval(self, query: str, context_docs: List[str], **kwargs) -> List[SearchResult]:
        """
        Perform retrieval with additional context documents.
        
        Args:
            query: Search query
            context_docs: Additional context documents
            **kwargs: Additional parameters
            
        Returns:
            Context-aware search results
        """
        try:
            # Combine query with context information
            context_keywords = []
            for doc in context_docs:
                keywords = self._extract_keywords(doc)
                context_keywords.extend(keywords)
            
            # Create enhanced query
            unique_keywords = list(set(context_keywords))[:5]
            enhanced_query = query + " " + " ".join(unique_keywords)
            
            # Perform search with enhanced query
            results = self.search_engine.search(enhanced_query, **kwargs)
            
            # Re-rank based on context relevance
            return self._rerank_by_context(results, context_docs)
            
        except Exception as e:
            self.logger.error(f"Contextual retrieval failed: {e}")
            raise SearchError(f"Contextual retrieval failed: {e}")
    
    def _rerank_by_context(self, results: List[SearchResult], context_docs: List[str]) -> List[SearchResult]:
        """
        Re-rank results based on context relevance.
        
        Args:
            results: Search results to re-rank
            context_docs: Context documents
            
        Returns:
            Re-ranked results
        """
        # Simple context-based re-ranking
        context_terms = set()
        for doc in context_docs:
            terms = self._extract_keywords(doc)
            context_terms.update(terms)
        
        # Calculate context relevance for each result
        for result in results:
            result_terms = set(self._extract_keywords(result.text))
            overlap = len(result_terms.intersection(context_terms))
            
            # Boost score based on context overlap
            context_boost = min(0.2, overlap * 0.05)  # Max 20% boost
            result.score += context_boost
        
        # Re-sort by updated scores
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
