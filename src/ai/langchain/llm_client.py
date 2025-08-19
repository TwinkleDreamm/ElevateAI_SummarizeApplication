from typing import List, Dict, Any, Optional

try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage, AIMessage
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore
    SystemMessage = None  # type: ignore
    HumanMessage = None  # type: ignore
    AIMessage = None  # type: ignore

from config.settings import settings


class LangchainLLMClient:
    """LLM client backed by LangChain (optional dependency).

    Designed to be drop-in compatible with existing `LLMClient.generate_response`
    call sites by returning a dict-like structure.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        # Initialize with settings from settings.py as defaults
        self.config = self._get_default_config()
        # Update with any provided config
        if config:
            self.config.update(config)
        self._init_llm()

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration and reinitialize LLM client."""
        self.config.update(new_config)
        self._init_llm()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration from settings.py."""
        return {
            "model_name": settings.openai_chat_model or "gpt-4o-mini",
            "temperature": getattr(settings, 'openai_temperature', 0.7),
            "max_tokens": getattr(settings, 'openai_max_tokens', 2000),
            "top_p": getattr(settings, 'openai_top_p', 0.9),
            "frequency_penalty": getattr(settings, 'openai_frequency_penalty', 0.0),
            "presence_penalty": getattr(settings, 'openai_presence_penalty', 0.0),
        }

    def _init_llm(self) -> None:
        if ChatOpenAI is None:
            self.llm = None
            return
        
        # Get configuration from settings with fallback to config, then defaults
        model_name = (self.config.get("model_name") or 
                     settings.openai_chat_model or 
                     "gpt-4o-mini")
        
        temperature = (self.config.get("temperature") or 
                     getattr(settings, 'openai_temperature', None) or 
                     0.7)
        
        max_tokens = (self.config.get("max_tokens") or 
                     getattr(settings, 'openai_max_tokens', None) or 
                     2000)
        
        top_p = (self.config.get("top_p") or 
                getattr(settings, 'openai_top_p', None) or 
                0.9)
        
        frequency_penalty = (self.config.get("frequency_penalty") or 
                           getattr(settings, 'openai_frequency_penalty', None) or 
                           0.0)
        
        presence_penalty = (self.config.get("presence_penalty") or 
                          getattr(settings, 'openai_presence_penalty', None) or 
                          0.0)
        
        # Initialize ChatOpenAI with custom base URL if available
        if settings.openai_api_key:
            # Use custom base URL from settings (for compatible providers like Ollama, LM Studio, etc.)
            if settings.openai_base_url and settings.openai_base_url != "https://api.openai.com/v1":
                self.llm = ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    openai_api_key=settings.openai_api_key,
                    openai_api_base=settings.openai_base_url
                )
            else:
                # Use standard OpenAI
                self.llm = ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    openai_api_key=settings.openai_api_key
                )
        else:
            # No API key configured
            self.llm = None

    def _build_llm_with_overrides(self, overrides: Dict[str, Any]):
        """Create a temporary LLM client with per-call overrides (e.g., max_tokens)."""
        if ChatOpenAI is None:
            return None
        params = {
            "model": (overrides.get("model_name") or self.config.get("model_name") or settings.openai_chat_model or "gpt-4o-mini"),
            "temperature": overrides.get("temperature", self.config.get("temperature", getattr(settings, 'openai_temperature', 0.7))),
            "max_tokens": overrides.get("max_tokens", self.config.get("max_tokens", getattr(settings, 'openai_max_tokens', 2000))),
            "top_p": overrides.get("top_p", self.config.get("top_p", getattr(settings, 'openai_top_p', 0.9))),
            "frequency_penalty": overrides.get("frequency_penalty", self.config.get("frequency_penalty", getattr(settings, 'openai_frequency_penalty', 0.0))),
            "presence_penalty": overrides.get("presence_penalty", self.config.get("presence_penalty", getattr(settings, 'openai_presence_penalty', 0.0))),
            "openai_api_key": settings.openai_api_key,
        }
        if settings.openai_base_url and settings.openai_base_url != "https://api.openai.com/v1":
            params["openai_api_base"] = settings.openai_base_url
        try:
            return ChatOpenAI(**params)
        except Exception:
            return self.llm

    def _to_langchain_messages(self, messages: List[Dict[str, str]]):
        if SystemMessage is None:
            return []
        result = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                result.append(SystemMessage(content=content))
            elif role in ("user", "human"):
                result.append(HumanMessage(content=content))
            elif role in ("assistant", "ai"): 
                result.append(AIMessage(content=content))
        return result

    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        if self.llm is None:
            # Check why LLM is not available
            if not settings.openai_api_key:
                return {
                    "content": "OpenAI API key not configured. Please set OPENAI_API_KEY in your environment.",
                    "model": self.config.get("model_name", "unknown"),
                    "usage": {},
                    "finish_reason": "no_api_key",
                }
            else:
                return {
                    "content": "LangChain is not available. Please install langchain and langchain-openai to enable this path.",
                    "model": self.config.get("model_name", "unknown"),
                    "usage": {},
                    "finish_reason": "fallback",
                }
        
        try:
            lc_messages = self._to_langchain_messages(messages)
            # Support per-call overrides (e.g., max_tokens for fast mode)
            overrides = {k: v for k, v in kwargs.items() if k in {"model_name", "temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"} and v is not None}
            llm_to_use = self._build_llm_with_overrides(overrides) if overrides else self.llm
            response = llm_to_use.invoke(lc_messages)
            
            # Get actual model name from response if available
            actual_model = getattr(response, "model", None) or self.config.get("model_name", "unknown")
            
            return {
                "content": response.content,
                "model": actual_model,
                "usage": getattr(response, "usage", {}),
                "finish_reason": getattr(response, "finish_reason", "stop"),
            }
        except Exception as e:  # pragma: no cover
            return {
                "content": f"LLM generation failed: {e}",
                "model": self.config.get("model_name", "unknown"),
                "usage": {},
                "finish_reason": "error",
            }

    def get_client_info(self) -> Dict[str, Any]:
        """Get information about the LLM client configuration."""
        return {
            "model_name": (self.config.get("model_name") or 
                          settings.openai_chat_model or 
                          "gpt-4o-mini"),
            "temperature": (self.config.get("temperature") or 
                          getattr(settings, 'openai_temperature', None) or 
                          0.7),
            "max_tokens": (self.config.get("max_tokens") or 
                          getattr(settings, 'openai_max_tokens', None) or 
                          2000),
            "top_p": (self.config.get("top_p") or 
                     getattr(settings, 'openai_top_p', None) or 
                     0.9),
            "frequency_penalty": (self.config.get("frequency_penalty") or 
                                getattr(settings, 'openai_frequency_penalty', None) or 
                                0.0),
            "presence_penalty": (self.config.get("presence_penalty") or 
                               getattr(settings, 'openai_presence_penalty', None) or 
                               0.0),
            "api_key_configured": bool(settings.openai_api_key),
            "base_url": settings.openai_base_url,
            "is_custom_provider": settings.openai_base_url != "https://api.openai.com/v1",
            "langchain_available": ChatOpenAI is not None,
            "llm_initialized": self.llm is not None
        }

    def test_connection(self) -> Dict[str, Any]:
        """Test the connection to the LLM service."""
        if not self.llm:
            return {
                "success": False,
                "error": "LLM client not initialized",
                "details": self.get_client_info()
            }
        
        try:
            # Simple test message
            test_message = [{"role": "user", "content": "Hello"}]
            response = self.generate_response(test_message)
            
            if response.get("finish_reason") in ["stop", "length"]:
                return {
                    "success": True,
                    "response": response["content"][:100] + "..." if len(response["content"]) > 100 else response["content"],
                    "model": response["model"],
                    "details": self.get_client_info()
                }
            else:
                return {
                    "success": False,
                    "error": f"Unexpected finish reason: {response.get('finish_reason')}",
                    "details": response
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "details": self.get_client_info()
            }

    def check_settings_applied(self) -> Dict[str, Any]:
        """Check if the current settings match the expected configuration."""
        current_config = self.config
        expected_config = {
            "model_name": settings.openai_chat_model,
            "temperature": getattr(settings, 'openai_temperature', None),
            "max_tokens": getattr(settings, 'openai_max_tokens', None),
            "top_p": getattr(settings, 'openai_top_p', None),
            "frequency_penalty": getattr(settings, 'openai_frequency_penalty', None),
            "presence_penalty": getattr(settings, 'openai_presence_penalty', None),
        }
        
        # Filter out None values
        expected_config = {k: v for k, v in expected_config.items() if v is not None}
        
        # Check which settings match
        matches = {}
        mismatches = {}
        
        for key, expected_value in expected_config.items():
            current_value = current_config.get(key)
            if current_value == expected_value:
                matches[key] = current_value
            else:
                mismatches[key] = {
                    "current": current_value,
                    "expected": expected_value
                }
        
        return {
            "settings_applied": len(mismatches) == 0,
            "settings_source": "settings.py",
            "matches": matches,
            "mismatches": mismatches,
            "current_config": current_config,
            "expected_config": expected_config
        }

    def debug_config(self) -> Dict[str, Any]:
        """Debug configuration state for troubleshooting."""
        return {
            "self_config": self.config,
            "settings_values": {
                "openai_chat_model": getattr(settings, 'openai_chat_model', 'NOT_SET'),
                "openai_temperature": getattr(settings, 'openai_temperature', 'NOT_SET'),
                "openai_max_tokens": getattr(settings, 'openai_max_tokens', 'NOT_SET'),
                "openai_top_p": getattr(settings, 'openai_top_p', 'NOT_SET'),
                "openai_frequency_penalty": getattr(settings, 'openai_frequency_penalty', 'NOT_SET'),
                "openai_presence_penalty": getattr(settings, 'openai_presence_penalty', 'NOT_SET'),
            },
            "default_config": self._get_default_config(),
            "llm_initialized": self.llm is not None,
            "llm_config": {
                "model": getattr(self.llm, 'model', 'N/A') if self.llm else 'N/A',
                "temperature": getattr(self.llm, 'temperature', 'N/A') if self.llm else 'N/A',
                "max_tokens": getattr(self.llm, 'max_tokens', 'N/A') if self.llm else 'N/A',
            } if self.llm else {}
        }


