"""
Reranking module for ElevateAI.
Improves search result ranking using various reranking strategies.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

from config.settings import settings
from src.utils.logger import logger
from src.utils.exceptions import SearchError
from .semantic_search import SearchResult


@dataclass
class RerankingScore:
    """Represents reranking scores for a result."""
    semantic_score: float
    lexical_score: float
    context_score: float
    freshness_score: float
    final_score: float


class Reranker:
    """Advanced reranking system for search results."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the reranker.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.settings = settings
        self.logger = logger
        
        # Reranking parameters
        self.use_cross_encoder = self.config.get('use_cross_encoder', False)
        self.cross_encoder_model = self.config.get(
            'cross_encoder_model', 
            'cross-encoder/ms-marco-MiniLM-L-6-v2'
        )
        
        # Scoring weights
        self.semantic_weight = self.config.get('semantic_weight', 0.4)
        self.lexical_weight = self.config.get('lexical_weight', 0.3)
        self.context_weight = self.config.get('context_weight', 0.2)
        self.freshness_weight = self.config.get('freshness_weight', 0.1)
        
        # Initialize cross-encoder if available and requested
        self.cross_encoder = None
        if self.use_cross_encoder and CROSS_ENCODER_AVAILABLE:
            self._load_cross_encoder()
    
    def _load_cross_encoder(self) -> None:
        """Load cross-encoder model for reranking."""
        try:
            self.logger.info(f"Loading cross-encoder model: {self.cross_encoder_model}")
            self.cross_encoder = CrossEncoder(self.cross_encoder_model)
            self.logger.info("Cross-encoder model loaded successfully")
        except Exception as e:
            self.logger.warning(f"Failed to load cross-encoder: {e}")
            self.cross_encoder = None
    
    def rerank_chunks(self, query: str, results: List[SearchResult], **kwargs) -> List[SearchResult]:
        """
        Rerank search results using multiple strategies.
        
        Args:
            query: Original search query
            results: Search results to rerank
            **kwargs: Additional reranking parameters
            
        Returns:
            Reranked search results
            
        Raises:
            SearchError: If reranking fails
        """
        if not results:
            return results
        
        try:
            self.logger.info(f"Reranking {len(results)} results for query: '{query}'")
            
            # Calculate various scores for each result
            reranked_results = []
            
            for result in results:
                # Calculate individual scores
                semantic_score = self._calculate_semantic_score(query, result)
                lexical_score = self._calculate_lexical_score(query, result)
                context_score = self._calculate_context_score(query, result, results)
                freshness_score = self._calculate_freshness_score(result)
                
                # Calculate final weighted score
                final_score = (
                    semantic_score * self.semantic_weight +
                    lexical_score * self.lexical_weight +
                    context_score * self.context_weight +
                    freshness_score * self.freshness_weight
                )
                
                # Create reranking score object
                reranking_score = RerankingScore(
                    semantic_score=semantic_score,
                    lexical_score=lexical_score,
                    context_score=context_score,
                    freshness_score=freshness_score,
                    final_score=final_score
                )
                
                # Update result with new score
                result.score = final_score
                result.metadata['reranking_scores'] = reranking_score
                
                reranked_results.append(result)
            
            # Sort by final score
            reranked_results.sort(key=lambda x: x.score, reverse=True)
            
            # Update ranks
            for i, result in enumerate(reranked_results):
                result.rank = i + 1
            
            # Apply cross-encoder reranking if available
            if self.cross_encoder and kwargs.get('use_cross_encoder', True):
                reranked_results = self._cross_encoder_rerank(query, reranked_results)
            
            self.logger.info(f"Reranking completed: {len(reranked_results)} results")
            return reranked_results
            
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            raise SearchError(f"Reranking failed: {e}")
    
    def _calculate_semantic_score(self, query: str, result: SearchResult) -> float:
        """
        Calculate semantic similarity score.
        
        Args:
            query: Search query
            result: Search result
            
        Returns:
            Semantic score (0-1)
        """
        # Use the original vector similarity score as base
        return min(1.0, max(0.0, result.score))
    
    def _calculate_lexical_score(self, query: str, result: SearchResult) -> float:
        """
        Calculate lexical matching score (BM25-like).
        
        Args:
            query: Search query
            result: Search result
            
        Returns:
            Lexical score (0-1)
        """
        try:
            query_terms = set(query.lower().split())
            result_terms = result.text.lower().split()
            result_term_set = set(result_terms)
            
            if not query_terms or not result_terms:
                return 0.0
            
            # Calculate term frequency scores
            tf_scores = []
            for term in query_terms:
                if term in result_term_set:
                    tf = result_terms.count(term)
                    # Simple TF normalization
                    tf_score = tf / (tf + 1.0)
                    tf_scores.append(tf_score)
                else:
                    tf_scores.append(0.0)
            
            # Calculate average TF score
            avg_tf = sum(tf_scores) / len(tf_scores) if tf_scores else 0.0
            
            # Calculate coverage (how many query terms are found)
            coverage = len([score for score in tf_scores if score > 0]) / len(query_terms)
            
            # Combine TF and coverage
            lexical_score = (avg_tf * 0.7) + (coverage * 0.3)
            
            return min(1.0, lexical_score)
            
        except Exception as e:
            self.logger.warning(f"Lexical score calculation failed: {e}")
            return 0.0
    
    def _calculate_context_score(self, query: str, result: SearchResult, all_results: List[SearchResult]) -> float:
        """
        Calculate context relevance score based on other results.
        
        Args:
            query: Search query
            result: Current result
            all_results: All search results for context
            
        Returns:
            Context score (0-1)
        """
        try:
            # Extract key terms from top results (excluding current)
            top_results = [r for r in all_results[:5] if r.id != result.id]
            
            if not top_results:
                return 0.5  # Neutral score if no context
            
            # Collect terms from top results
            context_terms = set()
            for top_result in top_results:
                terms = self._extract_terms(top_result.text)
                context_terms.update(terms)
            
            # Calculate overlap with current result
            result_terms = set(self._extract_terms(result.text))
            
            if not context_terms or not result_terms:
                return 0.5
            
            overlap = len(result_terms.intersection(context_terms))
            max_possible = min(len(result_terms), len(context_terms))
            
            if max_possible == 0:
                return 0.5
            
            context_score = overlap / max_possible
            return min(1.0, context_score)
            
        except Exception as e:
            self.logger.warning(f"Context score calculation failed: {e}")
            return 0.5
    
    def _calculate_freshness_score(self, result: SearchResult) -> float:
        """
        Calculate freshness score based on content age.
        
        Args:
            result: Search result
            
        Returns:
            Freshness score (0-1)
        """
        try:
            # Check if result has timestamp information
            metadata = result.metadata
            
            if 'added_at' in metadata:
                from datetime import datetime, timezone
                
                # Parse timestamp
                added_at = datetime.fromisoformat(metadata['added_at'].replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                
                # Calculate age in days
                age_days = (now - added_at).days
                
                # Fresher content gets higher score (exponential decay)
                freshness_score = np.exp(-age_days / 30.0)  # 30-day half-life
                return min(1.0, freshness_score)
            
            # Default score if no timestamp
            return 0.5
            
        except Exception as e:
            self.logger.warning(f"Freshness score calculation failed: {e}")
            return 0.5
    
    def _cross_encoder_rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """
        Rerank using cross-encoder model.
        
        Args:
            query: Search query
            results: Results to rerank
            
        Returns:
            Cross-encoder reranked results
        """
        if not self.cross_encoder or not results:
            return results
        
        try:
            # Prepare query-document pairs
            pairs = [(query, result.text) for result in results]
            
            # Get cross-encoder scores
            ce_scores = self.cross_encoder.predict(pairs)
            
            # Update results with cross-encoder scores
            for result, ce_score in zip(results, ce_scores):
                # Combine original score with cross-encoder score
                combined_score = (result.score * 0.6) + (float(ce_score) * 0.4)
                result.score = combined_score
                result.metadata['cross_encoder_score'] = float(ce_score)
            
            # Re-sort by combined score
            results.sort(key=lambda x: x.score, reverse=True)
            
            # Update ranks
            for i, result in enumerate(results):
                result.rank = i + 1
            
            self.logger.info("Cross-encoder reranking completed")
            return results
            
        except Exception as e:
            self.logger.warning(f"Cross-encoder reranking failed: {e}")
            return results
    
    def _extract_terms(self, text: str) -> List[str]:
        """
        Extract meaningful terms from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted terms
        """
        # Simple term extraction
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out stop words and short words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        terms = [word for word in words if word not in stop_words and len(word) > 2]
        return terms
    
    def get_reranking_explanation(self, result: SearchResult) -> Dict[str, Any]:
        """
        Get explanation of reranking scores for a result.
        
        Args:
            result: Search result
            
        Returns:
            Dictionary with reranking explanation
        """
        reranking_scores = result.metadata.get('reranking_scores')
        
        if not reranking_scores:
            return {"error": "No reranking scores available"}
        
        return {
            "final_score": reranking_scores.final_score,
            "components": {
                "semantic": {
                    "score": reranking_scores.semantic_score,
                    "weight": self.semantic_weight,
                    "contribution": reranking_scores.semantic_score * self.semantic_weight
                },
                "lexical": {
                    "score": reranking_scores.lexical_score,
                    "weight": self.lexical_weight,
                    "contribution": reranking_scores.lexical_score * self.lexical_weight
                },
                "context": {
                    "score": reranking_scores.context_score,
                    "weight": self.context_weight,
                    "contribution": reranking_scores.context_score * self.context_weight
                },
                "freshness": {
                    "score": reranking_scores.freshness_score,
                    "weight": self.freshness_weight,
                    "contribution": reranking_scores.freshness_score * self.freshness_weight
                }
            },
            "cross_encoder_score": result.metadata.get('cross_encoder_score')
        }
