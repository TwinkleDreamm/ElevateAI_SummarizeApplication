"""
Caching utilities for ElevateAI application.
"""
import hashlib
import pickle
import json
from pathlib import Path
from typing import Any, Optional, Dict, Union, Callable
from functools import wraps
import time
from datetime import datetime, timedelta

from config.settings import settings
from .logger import logger


class CacheManager:
    """Manages caching for various application components."""
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = Path(cache_dir) if cache_dir else settings.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache configuration
        self.default_ttl = 3600  # 1 hour
        self.max_cache_size = 1000  # Maximum number of cached items
        
        # In-memory cache for frequently accessed items
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Cache manager initialized with directory: {self.cache_dir}")
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        # Create a string representation of all arguments
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        
        # Create hash of the key data
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        # Check memory cache first
        if key in self.memory_cache:
            cache_entry = self.memory_cache[key]
            if self._is_cache_valid(cache_entry):
                return cache_entry['value']
            else:
                # Remove expired entry
                del self.memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.cache"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cache_entry = pickle.load(f)
                
                if self._is_cache_valid(cache_entry):
                    # Load into memory cache for faster access
                    self.memory_cache[key] = cache_entry
                    return cache_entry['value']
                else:
                    # Remove expired file
                    cache_file.unlink()
                    
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
                cache_file.unlink(missing_ok=True)
        
        return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        cache_entry = {
            'value': value,
            'expires_at': expires_at,
            'created_at': datetime.now()
        }
        
        # Store in memory cache
        self.memory_cache[key] = cache_entry
        
        # Clean up memory cache if too large
        if len(self.memory_cache) > self.max_cache_size:
            self._cleanup_memory_cache()
        
        # Store in disk cache
        cache_file = self.cache_dir / f"{key}.cache"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_entry, f)
        except Exception as e:
            logger.warning(f"Failed to save cache file {cache_file}: {e}")
    
    def delete(self, key: str) -> bool:
        """
        Delete item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if item was deleted
        """
        deleted = False
        
        # Remove from memory cache
        if key in self.memory_cache:
            del self.memory_cache[key]
            deleted = True
        
        # Remove from disk cache
        cache_file = self.cache_dir / f"{key}.cache"
        if cache_file.exists():
            cache_file.unlink()
            deleted = True
        
        return deleted
    
    def clear(self) -> None:
        """Clear all cache entries."""
        # Clear memory cache
        self.memory_cache.clear()
        
        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")
        
        logger.info("Cache cleared")
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        return datetime.now() < cache_entry['expires_at']
    
    def _cleanup_memory_cache(self) -> None:
        """Clean up memory cache by removing oldest entries."""
        # Sort by creation time and keep only the newest entries
        sorted_items = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1]['created_at'],
            reverse=True
        )
        
        # Keep only the newest 80% of entries
        keep_count = int(self.max_cache_size * 0.8)
        self.memory_cache = dict(sorted_items[:keep_count])
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        disk_files = list(self.cache_dir.glob("*.cache"))
        
        return {
            'memory_cache_size': len(self.memory_cache),
            'disk_cache_files': len(disk_files),
            'cache_directory': str(self.cache_dir),
            'total_disk_size': sum(f.stat().st_size for f in disk_files),
            'max_cache_size': self.max_cache_size,
            'default_ttl': self.default_ttl
        }


# Global cache manager instance
cache_manager = CacheManager()


def cached(ttl: Optional[int] = None, key_prefix: str = ""):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            func_name = f"{key_prefix}{func.__name__}" if key_prefix else func.__name__
            cache_key = f"{func_name}_{cache_manager._generate_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func_name}")
                return cached_result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for {func_name}, executing function")
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


class EmbeddingCache:
    """Specialized cache for embeddings."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize embedding cache."""
        self.cache_dir = cache_dir or settings.data_dir / "embedding_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_manager = CacheManager(self.cache_dir)
    
    def get_embedding(self, text: str, model_name: str) -> Optional[Any]:
        """Get cached embedding for text."""
        cache_key = self._get_embedding_key(text, model_name)
        return self.cache_manager.get(cache_key)
    
    def set_embedding(self, text: str, model_name: str, embedding: Any, ttl: int = 86400) -> None:
        """Cache embedding for text (default 24 hours)."""
        cache_key = self._get_embedding_key(text, model_name)
        self.cache_manager.set(cache_key, embedding, ttl)
    
    def _get_embedding_key(self, text: str, model_name: str) -> str:
        """Generate cache key for embedding."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"embedding_{model_name}_{text_hash}"


class LLMResponseCache:
    """Specialized cache for LLM responses."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize LLM response cache."""
        self.cache_dir = cache_dir or settings.data_dir / "llm_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_manager = CacheManager(self.cache_dir)
    
    def get_response(self, messages: list, model: str, temperature: float) -> Optional[str]:
        """Get cached LLM response."""
        cache_key = self._get_response_key(messages, model, temperature)
        return self.cache_manager.get(cache_key)
    
    def set_response(self, messages: list, model: str, temperature: float, 
                    response: str, ttl: int = 3600) -> None:
        """Cache LLM response (default 1 hour)."""
        cache_key = self._get_response_key(messages, model, temperature)
        self.cache_manager.set(cache_key, response, ttl)
    
    def _get_response_key(self, messages: list, model: str, temperature: float) -> str:
        """Generate cache key for LLM response."""
        key_data = {
            'messages': messages,
            'model': model,
            'temperature': temperature
        }
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"llm_response_{key_hash}"


# Global cache instances
embedding_cache = EmbeddingCache()
llm_cache = LLMResponseCache()
