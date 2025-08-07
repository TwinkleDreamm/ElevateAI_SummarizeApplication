"""
Prompt engineering module for ElevateAI.
Handles advanced prompt construction with role prompts, few-shot examples, and chain-of-thought.
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from config.settings import settings
from src.utils.logger import logger


class PromptType(Enum):
    """Types of prompts."""
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    ANALYSIS = "analysis"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"


@dataclass
class PromptTemplate:
    """Template for prompt construction."""
    system_prompt: str
    user_template: str
    few_shot_examples: List[Dict[str, str]]
    chain_of_thought: bool
    variables: List[str]


class PromptEngineer:
    """Advanced prompt engineering for various AI tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the prompt engineer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.settings = settings
        self.logger = logger
        
        # Load prompt templates
        self.templates = self._load_prompt_templates()
    
    def _load_prompt_templates(self) -> Dict[PromptType, PromptTemplate]:
        """Load predefined prompt templates."""
        templates = {}
        
        # Summarization template
        templates[PromptType.SUMMARIZATION] = PromptTemplate(
            system_prompt="""You are an expert content summarizer. Your task is to create comprehensive, accurate, and well-structured summaries of the provided content. 

Key requirements:
- Maintain factual accuracy and preserve important details
- Use clear, concise language appropriate for the target audience
- Structure the summary logically with main points and supporting details
- Preserve the original tone and context when possible
- Include key insights, conclusions, and actionable information""",
            
            user_template="""Please summarize the following content:

{content}

{additional_instructions}

Summary:""",
            
            few_shot_examples=[
                {
                    "input": "A 30-minute video about machine learning basics covering supervised learning, unsupervised learning, and neural networks with practical examples.",
                    "output": "**Machine Learning Basics Summary**\n\n**Main Topics Covered:**\n1. **Supervised Learning**: Uses labeled data to train models for prediction tasks\n2. **Unsupervised Learning**: Finds patterns in unlabeled data through clustering and dimensionality reduction\n3. **Neural Networks**: Computational models inspired by biological neurons, capable of learning complex patterns\n\n**Key Insights:**\n- Machine learning enables computers to learn from data without explicit programming\n- Different approaches suit different types of problems and data availability\n- Practical applications demonstrated across various industries\n\n**Duration**: 30 minutes of comprehensive content suitable for beginners"
                }
            ],
            
            chain_of_thought=True,
            variables=["content", "additional_instructions"]
        )
        
        # Question Answering template
        templates[PromptType.QUESTION_ANSWERING] = PromptTemplate(
            system_prompt="""You are a knowledgeable assistant that provides accurate, helpful answers based on the given context. 

Guidelines:
- Answer questions directly and comprehensively using only the provided context
- If information is not available in the context, clearly state this limitation
- Provide specific details and examples when available
- Maintain objectivity and cite relevant parts of the context
- If the question is ambiguous, ask for clarification""",
            
            user_template="""Context:
{context}

Question: {question}

Answer:""",
            
            few_shot_examples=[
                {
                    "input": "Context: The video discusses three types of machine learning: supervised, unsupervised, and reinforcement learning.\nQuestion: What are the main types of machine learning mentioned?",
                    "output": "Based on the provided context, there are three main types of machine learning mentioned:\n1. Supervised learning\n2. Unsupervised learning\n3. Reinforcement learning\n\nThese represent the fundamental categories of machine learning approaches discussed in the video."
                }
            ],
            
            chain_of_thought=False,
            variables=["context", "question"]
        )
        
        # Analysis template
        templates[PromptType.ANALYSIS] = PromptTemplate(
            system_prompt="""You are an expert analyst capable of deep content analysis. Your role is to examine content thoroughly and provide insightful analysis.

Analysis approach:
- Identify key themes, patterns, and insights
- Evaluate strengths, weaknesses, and opportunities
- Consider multiple perspectives and implications
- Provide evidence-based conclusions
- Suggest actionable recommendations when appropriate""",
            
            user_template="""Please analyze the following content:

{content}

Focus areas: {focus_areas}

Analysis:""",
            
            few_shot_examples=[],
            chain_of_thought=True,
            variables=["content", "focus_areas"]
        )
        
        return templates
    
    def build_prompt(self, prompt_type: PromptType, variables: Dict[str, Any], **kwargs) -> List[Dict[str, str]]:
        """
        Build a complete prompt with system message and user input.
        
        Args:
            prompt_type: Type of prompt to build
            variables: Variables to fill in the template
            **kwargs: Additional options
            
        Returns:
            List of message dictionaries for the LLM
        """
        if prompt_type not in self.templates:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        template = self.templates[prompt_type]
        
        # Build system message
        system_message = template.system_prompt
        
        # Add role-specific instructions if provided
        if 'role' in kwargs:
            role_instruction = f"\nAdditional role context: {kwargs['role']}"
            system_message += role_instruction
        
        # Build user message
        user_message = template.user_template.format(**variables)
        
        # Add few-shot examples if requested
        if kwargs.get('include_examples', True) and template.few_shot_examples:
            examples_text = self._format_few_shot_examples(template.few_shot_examples)
            user_message = examples_text + "\n\n" + user_message
        
        # Add chain-of-thought instruction if enabled
        if template.chain_of_thought and kwargs.get('use_chain_of_thought', True):
            cot_instruction = "\n\nPlease think through this step by step before providing your final answer."
            user_message += cot_instruction
        
        # Build message list
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        return messages
    
    def build_summarization_prompt(self, content: str, **kwargs) -> List[Dict[str, str]]:
        """
        Build a summarization prompt.
        
        Args:
            content: Content to summarize
            **kwargs: Additional options
            
        Returns:
            List of message dictionaries
        """
        variables = {
            "content": content,
            "additional_instructions": kwargs.get('instructions', '')
        }
        
        return self.build_prompt(PromptType.SUMMARIZATION, variables, **kwargs)
    
    def build_qa_prompt(self, context: str, question: str, **kwargs) -> List[Dict[str, str]]:
        """
        Build a question-answering prompt.
        
        Args:
            context: Context information
            question: Question to answer
            **kwargs: Additional options
            
        Returns:
            List of message dictionaries
        """
        variables = {
            "context": context,
            "question": question
        }
        
        return self.build_prompt(PromptType.QUESTION_ANSWERING, variables, **kwargs)
    
    def build_multi_hop_prompt(self, chunks: List[Dict[str, Any]], query: str, **kwargs) -> List[Dict[str, str]]:
        """
        Build a prompt for multi-hop reasoning across multiple chunks.
        
        Args:
            chunks: List of relevant chunks
            query: User query
            **kwargs: Additional options
            
        Returns:
            List of message dictionaries
        """
        # Combine chunks into context
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            chunk_text = chunk.get('text', chunk.get('content', ''))
            source_info = chunk.get('metadata', {}).get('source', f'Source {i}')
            context_parts.append(f"[{source_info}]\n{chunk_text}")
        
        combined_context = "\n\n".join(context_parts)
        
        # Add multi-hop specific instructions
        multi_hop_instructions = """
When answering, please:
1. Consider information from all provided sources
2. Synthesize information across different sources when relevant
3. Clearly indicate which sources support your answer
4. If sources contain conflicting information, acknowledge this
"""
        
        variables = {
            "context": combined_context,
            "question": query
        }
        
        # Build base QA prompt
        messages = self.build_prompt(PromptType.QUESTION_ANSWERING, variables, **kwargs)
        
        # Enhance system prompt for multi-hop reasoning
        messages[0]["content"] += multi_hop_instructions
        
        return messages
    
    def build_web_enhanced_prompt(self, local_chunks: List[Dict[str, Any]], 
                                 web_chunks: List[Dict[str, Any]], 
                                 query: str, **kwargs) -> List[Dict[str, str]]:
        """
        Build a prompt that combines local and web search results.
        
        Args:
            local_chunks: Local search results
            web_chunks: Web search results
            query: User query
            **kwargs: Additional options
            
        Returns:
            List of message dictionaries
        """
        # Separate local and web content
        local_context = []
        web_context = []
        
        for chunk in local_chunks:
            chunk_text = chunk.get('text', chunk.get('content', ''))
            source = chunk.get('metadata', {}).get('source', 'Local source')
            local_context.append(f"[Local: {source}]\n{chunk_text}")
        
        for chunk in web_chunks:
            chunk_text = chunk.get('text', chunk.get('content', ''))
            source = chunk.get('metadata', {}).get('source', 'Web source')
            web_context.append(f"[Web: {source}]\n{chunk_text}")
        
        # Combine contexts
        all_context = []
        if local_context:
            all_context.append("=== LOCAL SOURCES ===")
            all_context.extend(local_context)
        
        if web_context:
            all_context.append("\n=== WEB SOURCES ===")
            all_context.extend(web_context)
            all_context.append("\n⚠️ Note: Web sources may require verification")
        
        combined_context = "\n\n".join(all_context)
        
        # Web-specific instructions
        web_instructions = """
Important guidelines for web-enhanced responses:
- Prioritize information from local sources when available
- Clearly distinguish between local and web sources in your answer
- Include a note about web source reliability when using web information
- Cross-reference information between sources when possible
"""
        
        variables = {
            "context": combined_context,
            "question": query
        }
        
        messages = self.build_prompt(PromptType.QUESTION_ANSWERING, variables, **kwargs)
        messages[0]["content"] += web_instructions
        
        return messages
    
    def _format_few_shot_examples(self, examples: List[Dict[str, str]]) -> str:
        """Format few-shot examples for inclusion in prompts."""
        if not examples:
            return ""
        
        formatted_examples = ["Here are some examples:"]
        
        for i, example in enumerate(examples, 1):
            formatted_examples.append(f"\nExample {i}:")
            formatted_examples.append(f"Input: {example['input']}")
            formatted_examples.append(f"Output: {example['output']}")
        
        return "\n".join(formatted_examples)
    
    def optimize_prompt_length(self, messages: List[Dict[str, str]], max_tokens: int = 4000) -> List[Dict[str, str]]:
        """
        Optimize prompt length to fit within token limits.
        
        Args:
            messages: Original messages
            max_tokens: Maximum token limit
            
        Returns:
            Optimized messages
        """
        # Simple token estimation (4 chars per token)
        total_length = sum(len(msg["content"]) for msg in messages)
        estimated_tokens = total_length // 4
        
        if estimated_tokens <= max_tokens:
            return messages
        
        # Truncate user message content if needed
        target_length = max_tokens * 4 * 0.8  # Use 80% of limit
        
        optimized_messages = messages.copy()
        user_msg = next((msg for msg in optimized_messages if msg["role"] == "user"), None)
        
        if user_msg and len(user_msg["content"]) > target_length:
            truncated_content = user_msg["content"][:int(target_length)] + "...[truncated]"
            user_msg["content"] = truncated_content
            self.logger.warning("Prompt truncated to fit token limit")
        
        return optimized_messages
