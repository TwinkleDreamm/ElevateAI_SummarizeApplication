"""
LLM client module for ElevateAI.
Handles Azure OpenAI and OpenAI API interactions.
"""
import json
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass

try:
    from openai import AzureOpenAI, OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_openai import AzureChatOpenAI, ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from config.settings import settings
from src.utils.logger import logger
from src.utils.exceptions import LLMError
from src.utils.memory import memory_manager
from src.ai.function_calling import FunctionCaller


@dataclass
class LLMResponse:
    """Represents an LLM response."""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    function_calls: Optional[List[Dict[str, Any]]] = None


class LLMClient:
    """Client for interacting with Large Language Models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM client.
        
        Args:
            config: Optional configuration dictionary
        """
        if not OPENAI_AVAILABLE:
            raise LLMError("OpenAI library not available. Please install openai package.")
        
        self.config = config or {}
        self.settings = settings
        self.logger = logger
        
        # Model configuration
        self.use_azure = self.config.get('use_azure', True)
        self.model_name = self.config.get('model_name', settings.azure_openai_deployment_name)
        self.temperature = self.config.get('temperature', 0.7)
        self.max_tokens = self.config.get('max_tokens', 2000)
        
        # Initialize clients
        self.azure_client = None
        self.openai_client = None
        self.langchain_client = None
        
        # Initialize function caller
        self.function_caller = FunctionCaller()
        
        self._initialize_clients()
        
        # Fix: Log provider type after initialization
        provider_info = self.get_model_info()
        self.logger.info(f"LLM Client initialized - Provider: {provider_info['provider_type']}, Model: {self.model_name}")
    
    def _initialize_clients(self) -> None:
        """Initialize OpenAI clients."""
        try:
            # Fix: Detect provider type and initialize appropriate client
            if (self.use_azure and settings.azure_openai_api_key and 
                settings.azure_openai_endpoint):
                
                # Check if this is a third-party provider (like YesScale)
                if ("api.yescale.io" in settings.azure_openai_endpoint or
                    "api.openai.com" not in settings.azure_openai_endpoint and
                    "openai.azure.com" not in settings.azure_openai_endpoint):
                    
                    # Third-party provider - use OpenAI client with base_url
                    self.openai_client = OpenAI(
                        api_key=settings.azure_openai_api_key,
                        base_url=settings.azure_openai_endpoint
                    )
                    self.logger.info(f"Third-party LLM client initialized: {settings.azure_openai_endpoint}")
                    
                    # Initialize Langchain client for third-party
                    if LANGCHAIN_AVAILABLE:
                        self.langchain_client = ChatOpenAI(
                            api_key=settings.azure_openai_api_key,
                            base_url=settings.azure_openai_endpoint,
                            model_name=self.model_name,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens
                        )
                        self.logger.info("Langchain third-party client initialized")
                
                else:
                    # Real Azure OpenAI - use Azure client
                    self.azure_client = AzureOpenAI(
                        api_key=settings.azure_openai_api_key,
                        api_version=settings.azure_openai_api_version,
                        azure_endpoint=settings.azure_openai_endpoint
                    )
                    self.logger.info("Azure OpenAI client initialized")
                    
                    # Initialize Langchain Azure client if available
                    if LANGCHAIN_AVAILABLE:
                        self.langchain_client = AzureChatOpenAI(
                            api_key=settings.azure_openai_api_key,
                            api_version=settings.azure_openai_api_version,
                            azure_endpoint=settings.azure_openai_endpoint,
                            deployment_name=self.model_name,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens
                        )
                        self.logger.info("Langchain Azure client initialized")
            
            # Fallback to OpenAI client
            elif settings.openai_api_key:
                self.openai_client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
                self.logger.info("OpenAI client initialized")
                
                # Initialize Langchain OpenAI client if available
                if LANGCHAIN_AVAILABLE:
                    self.langchain_client = ChatOpenAI(
                        api_key=settings.openai_api_key,
                        model=self.model_name,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
            
            else:
                self.logger.warning("No OpenAI API credentials configured - LLM features will be disabled")

        except Exception as e:
            self.logger.error(f"Failed to initialize LLM clients: {e}")
            # Don't raise exception, just log the error
            pass
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """
        Generate response from LLM with memory integration.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse object

        Raises:
            LLMError: If generation fails
        """
        try:
            import time
            start_time = time.time()

            # Enhance messages with memory context if requested
            if kwargs.get('use_memory', True):
                messages = self._enhance_with_memory(messages, **kwargs)

            # Use Azure client if available
            if self.azure_client:
                response = self._generate_azure_response(messages, **kwargs)
            elif self.openai_client:
                response = self._generate_openai_response(messages, **kwargs)
            else:
                # Return a mock response when no client is available
                return LLMResponse(
                    content="LLM service not available - please configure API credentials",
                    model="none",
                    usage={'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                    function_calls=None
                )

            # Store conversation in memory if enabled
            if kwargs.get('store_in_memory', True) and len(messages) >= 2:
                processing_time = time.time() - start_time
                user_message = next((msg['content'] for msg in messages if msg['role'] == 'user'), '')

                # Calculate confidence score based on response characteristics
                confidence_score = self._calculate_confidence_score(response)

                memory_manager.add_conversation_turn(
                    user_input=user_message,
                    assistant_response=response.content,
                    context_used=kwargs.get('context_sources', []),
                    processing_time=processing_time,
                    confidence_score=confidence_score
                )

            return response

        except Exception as e:
            self.logger.error(f"LLM response generation failed: {e}")
            raise LLMError(f"LLM response generation failed: {e}")
    
    def _generate_azure_response(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response using Azure OpenAI."""
        try:
            # Fix: Use deployment_id for Azure OpenAI, not model
            deployment_id = kwargs.get('deployment_id', self.model_name)
            
            response = self.azure_client.chat.completions.create(
                model=deployment_id,  # This should be deployment_id for Azure
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                functions=kwargs.get('functions'),
                function_call=kwargs.get('function_call')
            )
            
            choice = response.choices[0]
            
            # Handle function calls
            function_calls = None
            if choice.message.function_call:
                function_calls = [{
                    'name': choice.message.function_call.name,
                    'arguments': choice.message.function_call.arguments
                }]
            
            # Check for empty content and provide fallback
            content = choice.message.content or ""
            if not content.strip():
                self.logger.warning("Empty content received from Azure model, providing fallback response")
                content = "I apologize, but I'm unable to generate a response at the moment. Please try rephrasing your question or check your configuration."
            
            return LLMResponse(
                content=content,
                model=response.model,
                usage=response.usage.model_dump() if response.usage else {},
                finish_reason=choice.finish_reason,
                function_calls=function_calls
            )
            
        except Exception as e:
            self.logger.error(f"Azure OpenAI generation failed: {e}")
            raise LLMError(f"Azure OpenAI generation failed: {e}")
    
    def _generate_openai_response(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response using OpenAI."""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                functions=kwargs.get('functions'),
                function_call=kwargs.get('function_call')
            )
            
            choice = response.choices[0]
            
            # Handle function calls
            function_calls = None
            if choice.message.function_call:
                function_calls = [{
                    'name': choice.message.function_call.name,
                    'arguments': choice.message.function_call.arguments
                }]
            
            # Check for empty content and provide fallback
            content = choice.message.content or ""
            if not content.strip():
                self.logger.warning("Empty content received from model, providing fallback response")
                content = "I apologize, but I'm unable to generate a response at the moment. Please try rephrasing your question or check your configuration."
            
            return LLMResponse(
                content=content,
                model=response.model,
                usage=response.usage.model_dump() if response.usage else {},
                finish_reason=choice.finish_reason,
                function_calls=function_calls
            )
            
        except Exception as e:
            self.logger.error(f"OpenAI generation failed: {e}")
            raise LLMError(f"OpenAI generation failed: {e}")
    
    def generate_streaming_response(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """
        Generate streaming response from LLM.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters
            
        Yields:
            Response chunks
        """
        try:
            # Fix: Use correct client and parameters based on provider type
            if self.azure_client:
                # Azure OpenAI - use deployment_id
                deployment_id = kwargs.get('deployment_id', self.model_name)
                response = self.azure_client.chat.completions.create(
                    model=deployment_id,
                    messages=messages,
                    temperature=kwargs.get('temperature', self.temperature),
                    max_tokens=kwargs.get('max_tokens', self.max_tokens),
                    stream=True
                )
            elif self.openai_client:
                # OpenAI or third-party - use model name
                model_name = kwargs.get('model', self.model_name)
                response = self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=kwargs.get('temperature', self.temperature),
                    max_tokens=kwargs.get('max_tokens', self.max_tokens),
                    stream=True
                )
            else:
                raise LLMError("No LLM client available")
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self.logger.error(f"Streaming response failed: {e}")
            raise LLMError(f"Streaming response failed: {e}")
    
    def generate_with_langchain(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate response using Langchain.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters
            
        Returns:
            Generated response text
        """
        if not LANGCHAIN_AVAILABLE or not self.langchain_client:
            raise LLMError("Langchain not available")
        
        try:
            # Convert messages to Langchain format
            lc_messages = []
            for msg in messages:
                if msg['role'] == 'system':
                    lc_messages.append(SystemMessage(content=msg['content']))
                elif msg['role'] == 'user':
                    lc_messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'assistant':
                    lc_messages.append(AIMessage(content=msg['content']))
            
            response = self.langchain_client.invoke(lc_messages)
            return response.content
            
        except Exception as e:
            self.logger.error(f"Langchain generation failed: {e}")
            raise LLMError(f"Langchain generation failed: {e}")
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    def validate_messages(self, messages: List[Dict[str, str]]) -> bool:
        """
        Validate message format.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            True if valid
        """
        if not messages:
            return False
        
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if 'role' not in msg or 'content' not in msg:
                return False
            if msg['role'] not in ['system', 'user', 'assistant']:
                return False
        
        return True
    
    def _enhance_with_memory(self, messages: List[Dict[str, str]], **kwargs) -> List[Dict[str, str]]:
        """
        Enhance messages with relevant memory context.

        Args:
            messages: Original messages
            **kwargs: Additional parameters

        Returns:
            Enhanced messages with memory context
        """
        try:
            # Get user query from messages
            user_query = ""
            for msg in reversed(messages):
                if msg['role'] == 'user':
                    user_query = msg['content']
                    break

            if not user_query:
                return messages

            # Get relevant context from memory
            max_context = kwargs.get('max_memory_context', 3)
            relevant_memories = memory_manager.get_relevant_context(user_query, max_context)
            conversation_context = memory_manager.get_conversation_context(max_turns=2)

            # Build context string
            context_parts = []

            # Add conversation history
            if conversation_context:
                context_parts.append("Recent conversation:")
                for turn in conversation_context[-2:]:  # Last 2 turns
                    context_parts.append(f"User: {turn.user_input}")
                    context_parts.append(f"Assistant: {turn.assistant_response[:200]}...")

            # Add relevant memories
            if relevant_memories:
                context_parts.append("\nRelevant context:")
                for memory in relevant_memories:
                    if memory.memory_type == 'fact':
                        context_parts.append(f"- {memory.content}")
                    elif memory.memory_type == 'conversation':
                        context_parts.append(f"- Previous discussion: {memory.content[:150]}...")

            # Enhance system message with context
            if context_parts:
                context_text = "\n".join(context_parts)
                enhanced_messages = messages.copy()

                # Find system message or create one
                system_msg_idx = None
                for i, msg in enumerate(enhanced_messages):
                    if msg['role'] == 'system':
                        system_msg_idx = i
                        break

                if system_msg_idx is not None:
                    # Enhance existing system message
                    enhanced_messages[system_msg_idx]['content'] += f"\n\nContext from previous interactions:\n{context_text}"
                else:
                    # Add new system message
                    system_msg = {
                        'role': 'system',
                        'content': f"You are a helpful AI assistant. Use the following context from previous interactions when relevant:\n\n{context_text}"
                    }
                    enhanced_messages.insert(0, system_msg)

                return enhanced_messages

            return messages

        except Exception as e:
            self.logger.warning(f"Failed to enhance messages with memory: {e}")
            return messages

    def _calculate_confidence_score(self, response: LLMResponse) -> float:
        """
        Calculate confidence score for a response.

        Args:
            response: LLM response

        Returns:
            Confidence score between 0 and 1
        """
        base_score = 0.7

        # Adjust based on finish reason
        if response.finish_reason == 'stop':
            base_score += 0.2
        elif response.finish_reason == 'length':
            base_score -= 0.1

        # Adjust based on response length
        if response.content:
            word_count = len(response.content.split())
            if 20 <= word_count <= 500:
                base_score += 0.1
            elif word_count < 10:
                base_score -= 0.2

        return min(1.0, max(0.0, base_score))

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Dictionary with memory statistics
        """
        return memory_manager.get_memory_stats()

    def clear_conversation_memory(self) -> None:
        """Clear short-term conversation memory."""
        memory_manager.short_term.clear_context()
        self.logger.info("Conversation memory cleared")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        info = {
            'provider_type': self._get_provider_type(),
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'azure_client_available': self.azure_client is not None,
            'openai_client_available': self.openai_client is not None,
            'langchain_client_available': self.langchain_client is not None
        }
        
        # Add provider-specific information
        if self.azure_client:
            info['azure_endpoint'] = getattr(settings, 'azure_openai_endpoint', 'Not configured')
        elif self.openai_client:
            info['base_url'] = getattr(settings, 'openai_api_base', 'https://api.openai.com/v1')
        
        return info
    
    def _get_provider_type(self) -> str:
        """Get the type of provider being used."""
        if self.azure_client:
            return "Azure OpenAI"
        elif self.openai_client:
            if hasattr(settings, 'azure_openai_endpoint') and settings.azure_openai_endpoint:
                if "api.yescale.io" in settings.azure_openai_endpoint:
                    return "YesScale (Third-party)"
                elif "openai.azure.com" not in settings.azure_openai_endpoint:
                    return "Third-party Provider"
            return "OpenAI"
        else:
            return "None"
    
    # Function Calling Methods
    def get_available_functions(self) -> List[str]:
        """Get list of available function names."""
        return self.function_caller.get_available_functions()
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get function definitions in OpenAI format for LLM."""
        return self.function_caller.get_function_definitions()
    
    def register_function(self, name: str, description: str, parameters: Dict[str, Any], 
                         function: callable) -> None:
        """Register a new function for calling."""
        self.function_caller.register_function(name, description, parameters, function)
    
    def call_function(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific function by name with arguments."""
        return self.function_caller.call_function(name, arguments)
    
    def process_function_calls(self, function_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process function calls and return results."""
        return self.function_caller.process_function_calls(function_calls)
    
    def generate_response_with_functions(self, messages: List[Dict[str, str]], 
                                       use_functions: bool = True, **kwargs) -> LLMResponse:
        """
        Generate response with optional function calling support.
        
        Args:
            messages: List of message dictionaries
            use_functions: Whether to enable function calling
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with potential function calls
        """
        try:
            self.logger.info(f"Generating response with functions: {use_functions}")
            self.logger.info(f"Messages count: {len(messages)}")
            
            if use_functions:
                # Add function definitions to the request
                functions = self.get_function_definitions()
                self.logger.info(f"Available functions: {len(functions)}")
                
                if functions:
                    kwargs['functions'] = functions
                    kwargs['function_call'] = 'auto'  # Let the model decide when to call functions
                    self.logger.info("Function definitions added to request")
                else:
                    self.logger.warning("No function definitions available")
            
            # Generate response
            self.logger.info("Calling generate_response...")
            response = self.generate_response(messages, **kwargs)
            self.logger.info(f"Response received: {response.content[:100]}...")
            
            # If response contains function calls, process them
            if response.function_calls:
                self.logger.info(f"Processing {len(response.function_calls)} function calls")
                function_results = self.process_function_calls(response.function_calls)
                
                # Add function results to messages for follow-up
                function_message = self.function_caller.create_function_call_message(function_results)
                messages.append(function_message)
                
                # Generate final response with function results
                self.logger.info("Generating final response with function results...")
                final_response = self.generate_response(messages, **kwargs)
                self.logger.info(f"Final response: {final_response.content[:100]}...")
                return final_response
            
            self.logger.info("No function calls detected, returning original response")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in generate_response_with_functions: {e}")
            raise
