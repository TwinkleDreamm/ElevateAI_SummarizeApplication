"""
Summarization module for ElevateAI.
Handles intelligent content summarization with various strategies.
"""
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

from config.settings import settings
from src.utils.logger import logger
from src.utils.exceptions import LLMError
from .llm_client import LLMClient
from .prompt_engineer import PromptEngineer, PromptType


class SummaryType(Enum):
    """Types of summaries."""
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    BULLET_POINTS = "bullet_points"
    STRUCTURED = "structured"
    EXECUTIVE = "executive"


@dataclass
class SummaryResult:
    """Result of summarization."""
    summary: str
    summary_type: SummaryType
    source_info: Dict[str, Any]
    confidence_score: float
    word_count: int
    compression_ratio: float


class Summarizer:
    """Intelligent content summarizer."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the summarizer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.settings = settings
        self.logger = logger
        
        # Initialize components
        self.llm_client = LLMClient(config)
        self.prompt_engineer = PromptEngineer(config)
        
        # Summarization parameters
        self.default_max_length = self.config.get('max_length', 500)
        self.default_min_length = self.config.get('min_length', 100)
        self.temperature = self.config.get('temperature', 0.3)  # Lower for more focused summaries
    
    def summarize_chunks(self, chunks: List[Dict[str, Any]], query: Optional[str] = None, 
                        **kwargs) -> SummaryResult:
        """
        Summarize multiple chunks of content.
        
        Args:
            chunks: List of content chunks
            query: Optional query for focused summarization
            **kwargs: Additional summarization options
            
        Returns:
            SummaryResult object
            
        Raises:
            LLMError: If summarization fails
        """
        if not chunks:
            raise LLMError("No chunks provided for summarization")
        
        try:
            self.logger.info(f"Summarizing {len(chunks)} chunks")
            
            # Determine summarization strategy
            summary_type = kwargs.get('summary_type', SummaryType.ABSTRACTIVE)
            
            # Combine chunks into content
            combined_content = self._combine_chunks(chunks)
            
            # Check if content needs to be split for processing
            if len(combined_content) > 8000:  # Rough token limit check
                return self._summarize_long_content(chunks, query, **kwargs)
            else:
                return self._summarize_single_pass(combined_content, chunks, query, **kwargs)
                
        except Exception as e:
            self.logger.error(f"Summarization failed: {e}")
            raise LLMError(f"Summarization failed: {e}")
    
    def _combine_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Combine chunks into a single content string."""
        content_parts = []
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get('text', chunk.get('content', ''))
            source_info = chunk.get('metadata', {}).get('source', f'Source {i+1}')
            
            # Add source attribution
            content_parts.append(f"[{source_info}]\n{chunk_text}")
        
        return "\n\n".join(content_parts)
    
    def _summarize_single_pass(self, content: str, chunks: List[Dict[str, Any]], 
                              query: Optional[str] = None, **kwargs) -> SummaryResult:
        """Summarize content in a single pass."""
        summary_type = kwargs.get('summary_type', SummaryType.ABSTRACTIVE)
        
        # Build appropriate prompt
        if query:
            instructions = f"Focus on information relevant to: {query}"
        else:
            instructions = "Provide a comprehensive summary of the main points."
        # Allow callers to append extra guidance such as target language or length
        extra_instructions = kwargs.get('extra_instructions') or kwargs.get('language_instruction')
        if extra_instructions:
            instructions += f"\n{extra_instructions}"
        
        # Add summary type specific instructions
        instructions += self._get_type_specific_instructions(summary_type)
        
        # Build prompt
        messages = self.prompt_engineer.build_summarization_prompt(
            content=content,
            instructions=instructions,
            use_chain_of_thought=kwargs.get('use_cot', True)
        )
        
        # Generate summary
        response = self.llm_client.generate_response(
            messages,
            temperature=self.temperature,
            max_tokens=kwargs.get('max_tokens', 1000)
        )
        
        # Calculate metrics
        original_word_count = len(content.split())
        summary_word_count = len(response.content.split())
        compression_ratio = summary_word_count / original_word_count if original_word_count > 0 else 0
        
        # Extract source information
        self.logger.info("Start parallel extract source info")
        source_info = self._extract_source_info(chunks)
        self.logger.info("Finished parallel extract source info")

        return SummaryResult(
            summary=response.content,
            summary_type=summary_type,
            source_info=source_info,
            confidence_score=self._calculate_confidence_score(response),
            word_count=summary_word_count,
            compression_ratio=compression_ratio
        )
    
    def _summarize_long_content(self, chunks: List[Dict[str, Any]], 
                               query: Optional[str] = None, **kwargs) -> SummaryResult:
        """Summarize long content using hierarchical approach with parallel processing."""
        from .batch_processor import BatchProcessor
        
        self.logger.info("Using parallel hierarchical summarization for long content")
        
        # Split chunks into groups
        chunk_groups = self._group_chunks(chunks, max_group_size=5)
        
        # Initialize batch processor for parallel processing
        batch_processor = BatchProcessor(max_workers=min(4, len(chunk_groups)))
        
        def process_chunk_group(group: List[Dict[str, Any]], group_index: int) -> str:
            """Process a single chunk group."""
            group_content = self._combine_chunks(group)
            
            # Create focused prompt for intermediate summary
            instructions = f"Summarize the key points from this section (Part {group_index + 1})."
            if query:
                instructions += f" Focus on information relevant to: {query}"
            
            messages = self.prompt_engineer.build_summarization_prompt(
                content=group_content,
                instructions=instructions
            )
            
            response = self.llm_client.generate_response(
                messages,
                temperature=self.temperature,
                max_tokens=300
            )
            
            return response.content
        
        # Process chunk groups in parallel while maintaining order
        intermediate_summaries = batch_processor.process_batches(chunk_groups, process_chunk_group)
        
        # Combine intermediate summaries into final summary
        combined_intermediate = "\n\n".join([
            f"Section {i+1}: {summary}" 
            for i, summary in enumerate(intermediate_summaries)
        ])
        
        # Final summarization pass
        final_instructions = "Synthesize the following section summaries into a comprehensive final summary."
        if query:
            final_instructions += f" Focus on information relevant to: {query}"
        self.logger.info("Start build_summarization_prompt")
        final_messages = self.prompt_engineer.build_summarization_prompt(
            content=combined_intermediate,
            instructions=final_instructions
        )
        self.logger.info("Finished build_summarization_prompt")
        self.logger.info("Start generate_response")
        final_response = self.llm_client.generate_response(
            final_messages,
            temperature=self.temperature,
            max_tokens=kwargs.get('max_tokens', 800)
        )
        self.logger.info("Finished generate_response")
        # Calculate metrics
        total_original_words = sum(len(chunk.get('text', '').split()) for chunk in chunks)
        summary_word_count = len(final_response.content.split())
        compression_ratio = summary_word_count / total_original_words if total_original_words > 0 else 0
        
        self.logger.info("Start parallel extract source info")
        source_info = self._extract_source_info(chunks)
        self.logger.info("Finished parallel extract source info")
        
        return SummaryResult(
            summary=final_response.content,
            summary_type=kwargs.get('summary_type', SummaryType.ABSTRACTIVE),
            source_info=source_info,
            confidence_score=self._calculate_confidence_score(final_response),
            word_count=summary_word_count,
            compression_ratio=compression_ratio
        )
    
    def finalize_summary(self, intermediate_summaries: List[str], **kwargs) -> str:
        """
        Finalize summary from intermediate summaries.
        
        Args:
            intermediate_summaries: List of intermediate summaries
            **kwargs: Additional options
            
        Returns:
            Final summary
        """
        if not intermediate_summaries:
            return ""
        
        if len(intermediate_summaries) == 1:
            return intermediate_summaries[0]
        
        # Combine intermediate summaries
        combined_content = "\n\n".join([
            f"Summary {i+1}:\n{summary}"
            for i, summary in enumerate(intermediate_summaries)
        ])
        
        # Create final synthesis prompt
        instructions = "Synthesize the following summaries into a single, coherent final summary. Remove redundancy and ensure logical flow."
        
        messages = self.prompt_engineer.build_summarization_prompt(
            content=combined_content,
            instructions=instructions
        )
        
        response = self.llm_client.generate_response(
            messages,
            temperature=self.temperature,
            max_tokens=kwargs.get('max_tokens', 600)
        )
        
        return response.content
    
    def _get_type_specific_instructions(self, summary_type: SummaryType) -> str:
        """Get instructions specific to summary type."""
        instructions = {
            SummaryType.EXTRACTIVE: "\nUse exact phrases and sentences from the original content.",
            SummaryType.ABSTRACTIVE: "\nUse your own words to capture the essence of the content.",
            SummaryType.BULLET_POINTS: "\nFormat the summary as clear, concise bullet points.",
            SummaryType.STRUCTURED: "\nOrganize the summary with clear headings and sections.",
            SummaryType.EXECUTIVE: "\nProvide an executive summary suitable for decision-makers."
        }
        
        return instructions.get(summary_type, "")
    
    def _group_chunks(self, chunks: List[Dict[str, Any]], max_group_size: int = 5) -> List[List[Dict[str, Any]]]:
        """Group chunks for hierarchical processing."""
        groups = []
        current_group = []
        
        for chunk in chunks:
            current_group.append(chunk)
            
            if len(current_group) >= max_group_size:
                groups.append(current_group)
                current_group = []
        
        # Add remaining chunks
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _extract_source_info(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract source information from chunks in parallel."""
        from .batch_processor import BatchProcessor
        
        total_chunks = len(chunks)
        if total_chunks < 100:  # Use regular processing for small number of chunks
            return self._extract_source_info_sequential(chunks)
        
        # Initialize batch processor with appropriate number of workers
        batch_processor = BatchProcessor(max_workers=min(4, total_chunks // 25))
        
        def process_chunk_metadata(chunk: Dict[str, Any], _: int) -> Dict[str, set]:
            """Process metadata from a single chunk."""
            metadata = chunk.get('metadata', {})
            return {
                'sources': {metadata['source']} if 'source' in metadata else set(),
                'content_types': {metadata['content_type']} if 'content_type' in metadata else set()
            }
        
        # Process chunks in parallel
        chunk_results = batch_processor.process_batches(chunks, process_chunk_metadata)
        
        # Combine results
        all_sources = set()
        all_content_types = set()
        
        for result in chunk_results:
            all_sources.update(result['sources'])
            all_content_types.update(result['content_types'])
        
        return {
            'sources': list(all_sources),
            'content_types': list(all_content_types),
            'total_chunks': total_chunks,
            'has_web_content': any('web' in str(source).lower() for source in all_sources)
        }
        
    def _extract_source_info_sequential(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract source information from chunks sequentially."""
        sources = set()
        content_types = set()
        total_chunks = len(chunks)
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            if 'source' in metadata:
                sources.add(metadata['source'])
            if 'content_type' in metadata:
                content_types.add(metadata['content_type'])
        
        return {
            'sources': list(sources),
            'content_types': list(content_types),
            'total_chunks': total_chunks,
            'has_web_content': any('web' in str(source).lower() for source in sources)
        }
    
    def _calculate_confidence_score(self, response) -> float:
        """Calculate confidence score for summary."""
        # Simple heuristic based on response characteristics
        base_score = 0.8
        
        # Adjust based on finish reason
        if hasattr(response, 'finish_reason'):
            if response.finish_reason == 'stop':
                base_score += 0.1
            elif response.finish_reason == 'length':
                base_score -= 0.1
        
        # Adjust based on content length
        if hasattr(response, 'content'):
            word_count = len(response.content.split())
            if 50 <= word_count <= 500:  # Reasonable summary length
                base_score += 0.1
            elif word_count < 20:  # Too short
                base_score -= 0.2
        
        return min(1.0, max(0.0, base_score))
    
    def get_summary_statistics(self, summary_result: SummaryResult) -> Dict[str, Any]:
        """
        Get statistics about a summary.
        
        Args:
            summary_result: Summary result to analyze
            
        Returns:
            Dictionary with summary statistics
        """
        return {
            'word_count': summary_result.word_count,
            'compression_ratio': summary_result.compression_ratio,
            'confidence_score': summary_result.confidence_score,
            'summary_type': summary_result.summary_type.value,
            'source_count': len(summary_result.source_info.get('sources', [])),
            'has_web_content': summary_result.source_info.get('has_web_content', False),
            'total_source_chunks': summary_result.source_info.get('total_chunks', 0)
        }
