"""
Semantic search engine for ElevateAI.
Handles vector-based similarity search with advanced filtering.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from config.settings import settings
from src.utils.logger import logger
from src.utils.exceptions import SearchError
from src.database.vector_database import VectorDatabase


@dataclass
class SearchResult:
    """Represents a search result."""
    id: int
    score: float
    text: str
    metadata: Dict[str, Any]
    rank: int


class SemanticSearchEngine:
    """Advanced semantic search engine with filtering and ranking."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the semantic search engine.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.settings = settings
        self.logger = logger
        
        # Search parameters
        self.similarity_threshold = self.config.get(
            'similarity_threshold', 
            settings.similarity_threshold
        )
        self.max_results = self.config.get('max_results', 50)
        self.rerank_top_k = self.config.get('rerank_top_k', settings.rerank_top_k)
        
        # Initialize vector database
        self.vector_db = VectorDatabase(config)
    
    def search(self, query: str, **kwargs) -> List[SearchResult]:
        """
        Perform semantic search.
        
        Args:
            query: Search query
            **kwargs: Additional search parameters
            
        Returns:
            List of search results
            
        Raises:
            SearchError: If search fails
        """
        if not query or not query.strip():
            return []
        
        try:
            # Extract search parameters
            k = kwargs.get('k', self.max_results)
            threshold = kwargs.get('threshold', self.similarity_threshold)
            filters = kwargs.get('filters', {})
            
            self.logger.info(f"Searching for: '{query}' (k={k}, threshold={threshold})")
            
            # Perform vector search
            raw_results = self.vector_db.search(query, k=k, threshold=threshold)
            
            # Apply metadata filters
            if filters:
                raw_results = self._apply_filters(raw_results, filters)
            
            # Convert to SearchResult objects
            search_results = []
            for i, result in enumerate(raw_results):
                search_result = SearchResult(
                    id=result['id'],
                    score=result['score'],
                    text=result['metadata'].get('text', ''),
                    metadata=result['metadata'],
                    rank=i + 1
                )
                search_results.append(search_result)
            
            self.logger.info(f"Found {len(search_results)} results")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise SearchError(f"Search failed: {e}")
    
    def multi_query_search(self, queries: List[str], **kwargs) -> List[SearchResult]:
        """
        Perform search with multiple queries and combine results.
        
        Args:
            queries: List of search queries
            **kwargs: Additional search parameters
            
        Returns:
            Combined and deduplicated search results
        """
        if not queries:
            return []
        
        try:
            all_results = {}  # Use dict to deduplicate by ID
            
            for query in queries:
                results = self.search(query, **kwargs)
                
                for result in results:
                    if result.id not in all_results:
                        all_results[result.id] = result
                    else:
                        # Keep the result with higher score
                        if result.score > all_results[result.id].score:
                            all_results[result.id] = result
            
            # Sort by score and re-rank
            combined_results = sorted(
                all_results.values(),
                key=lambda x: x.score,
                reverse=True
            )
            
            # Update ranks
            for i, result in enumerate(combined_results):
                result.rank = i + 1
            
            return combined_results
            
        except Exception as e:
            self.logger.error(f"Multi-query search failed: {e}")
            raise SearchError(f"Multi-query search failed: {e}")
    
    def search_with_expansion(self, query: str, **kwargs) -> List[SearchResult]:
        """
        Search with query expansion using synonyms and related terms.
        
        Args:
            query: Original search query
            **kwargs: Additional search parameters
            
        Returns:
            Search results from expanded queries
        """
        try:
            # Generate expanded queries (simplified approach)
            expanded_queries = self._expand_query(query)
            
            # Perform multi-query search
            return self.multi_query_search(expanded_queries, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Search with expansion failed: {e}")
            raise SearchError(f"Search with expansion failed: {e}")
    
    def search_by_similarity(self, reference_text: str, **kwargs) -> List[SearchResult]:
        """
        Find similar content to a reference text.
        
        Args:
            reference_text: Reference text to find similar content
            **kwargs: Additional search parameters
            
        Returns:
            List of similar search results
        """
        try:
            # Generate embedding for reference text
            embedding = self.vector_db.embedding_generator.generate_embedding(reference_text)
            
            # Search using the embedding
            k = kwargs.get('k', self.max_results)
            threshold = kwargs.get('threshold', self.similarity_threshold)
            
            raw_results = self.vector_db.search_by_vector(embedding, k=k, threshold=threshold)
            
            # Apply filters if provided
            filters = kwargs.get('filters', {})
            if filters:
                raw_results = self._apply_filters(raw_results, filters)
            
            # Convert to SearchResult objects
            search_results = []
            for i, result in enumerate(raw_results):
                search_result = SearchResult(
                    id=result['id'],
                    score=result['score'],
                    text=result['metadata'].get('text', ''),
                    metadata=result['metadata'],
                    rank=i + 1
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            raise SearchError(f"Similarity search failed: {e}")
    
    def _apply_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply metadata filters to search results.
        
        Args:
            results: Raw search results
            filters: Filter criteria
            
        Returns:
            Filtered results
        """
        filtered_results = []
        
        for result in results:
            metadata = result.get('metadata', {})
            matches = True
            
            for key, value in filters.items():
                if key not in metadata:
                    matches = False
                    break
                
                # Handle different filter types
                if isinstance(value, dict):
                    # Range or comparison filters
                    if '$gte' in value and metadata[key] < value['$gte']:
                        matches = False
                        break
                    if '$lte' in value and metadata[key] > value['$lte']:
                        matches = False
                        break
                    if '$in' in value and metadata[key] not in value['$in']:
                        matches = False
                        break
                    if '$ne' in value and metadata[key] == value['$ne']:
                        matches = False
                        break
                elif isinstance(value, str) and isinstance(metadata[key], str):
                    # String contains search (case-insensitive)
                    if value.lower() not in metadata[key].lower():
                        matches = False
                        break
                else:
                    # Exact match
                    if metadata[key] != value:
                        matches = False
                        break
            
            if matches:
                filtered_results.append(result)
        
        return filtered_results
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms and related terms (simplified).
        
        Args:
            query: Original query
            
        Returns:
            List of expanded queries
        """
        # This is a simplified implementation
        # In a real system, you might use WordNet, word embeddings, or LLMs
        
        expanded_queries = [query]
        
        # Simple synonym mapping (can be extended)
        synonyms = {
            'video': ['clip', 'recording', 'footage'],
            'audio': ['sound', 'recording', 'speech'],
            'document': ['text', 'file', 'paper'],
            'summary': ['overview', 'abstract', 'synopsis'],
            'analysis': ['examination', 'study', 'review']
        }
        
        query_words = query.lower().split()
        
        for word in query_words:
            if word in synonyms:
                for synonym in synonyms[word]:
                    # Create new query by replacing the word with synonym
                    new_query = query.lower().replace(word, synonym)
                    if new_query not in expanded_queries:
                        expanded_queries.append(new_query)
        
        return expanded_queries[:5]  # Limit to 5 queries to avoid too many requests
    
    def get_search_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """
        Get search suggestions based on partial query.
        
        Args:
            partial_query: Partial search query
            limit: Maximum number of suggestions
            
        Returns:
            List of search suggestions
        """
        try:
            # This is a simplified implementation
            # In practice, you might use a dedicated search suggestion system
            
            if len(partial_query) < 2:
                return []
            
            # Search for similar content and extract common terms
            results = self.search(partial_query, k=20, threshold=0.3)
            
            suggestions = set()
            for result in results:
                text = result.text.lower()
                words = text.split()
                
                # Find phrases that start with the partial query
                for i, word in enumerate(words):
                    if word.startswith(partial_query.lower()):
                        # Add the word and potentially the next few words
                        phrase = word
                        if i + 1 < len(words):
                            phrase += " " + words[i + 1]
                        suggestions.add(phrase)
                        
                        if len(suggestions) >= limit:
                            break
                
                if len(suggestions) >= limit:
                    break
            
            return list(suggestions)[:limit]
            
        except Exception as e:
            self.logger.warning(f"Failed to get search suggestions: {e}")
            return []
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """
        Get search engine statistics.
        
        Returns:
            Dictionary with search statistics
        """
        try:
            db_stats = self.vector_db.get_database_stats()
            
            return {
                'total_documents': db_stats['total_vectors'],
                'embedding_dimension': db_stats['embedding_dimension'],
                'index_type': db_stats['index_type'],
                'similarity_threshold': self.similarity_threshold,
                'max_results': self.max_results,
                'database_path': db_stats['database_path']
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to get search statistics: {e}")
            return {}
    
    def check_relevant_chunks(self, results: List[SearchResult], threshold: float = None) -> bool:
        """
        Check if search results contain relevant chunks above threshold.
        
        Args:
            results: Search results to check
            threshold: Relevance threshold (uses default if None)
            
        Returns:
            True if relevant chunks found
        """
        if not results:
            return False
        
        threshold = threshold or self.similarity_threshold
        
        relevant_count = sum(1 for result in results if result.score >= threshold)
        
        return relevant_count > 0
