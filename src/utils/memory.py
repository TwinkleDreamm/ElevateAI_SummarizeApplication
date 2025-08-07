"""
Memory management system for ElevateAI.
Implements short-term and long-term memory for conversation context.
"""
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import threading
from collections import deque

from config.settings import settings
from .logger import logger
from .cache import cache_manager


@dataclass
class MemoryEntry:
    """Represents a single memory entry."""
    id: str
    content: str
    memory_type: str  # 'conversation', 'fact', 'context', 'summary'
    importance: float  # 0.0 to 1.0
    timestamp: datetime
    source: str
    metadata: Dict[str, Any]
    access_count: int = 0
    last_accessed: Optional[datetime] = None


@dataclass
class ConversationTurn:
    """Represents a single conversation turn."""
    turn_id: str
    user_input: str
    assistant_response: str
    context_used: List[str]
    timestamp: datetime
    processing_time: float
    confidence_score: float


class ShortTermMemory:
    """Manages short-term memory for active conversations."""
    
    def __init__(self, max_size: int = 50, ttl_minutes: int = 30):
        """
        Initialize short-term memory.
        
        Args:
            max_size: Maximum number of entries to keep
            ttl_minutes: Time to live for entries in minutes
        """
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
        self.entries: deque = deque(maxlen=max_size)
        self.conversation_history: List[ConversationTurn] = []
        self.current_context: Dict[str, Any] = {}
        self.lock = threading.Lock()
        
        logger.info(f"Short-term memory initialized: max_size={max_size}, ttl={ttl_minutes}min")
    
    def add_entry(self, entry: MemoryEntry) -> None:
        """Add entry to short-term memory."""
        with self.lock:
            # Remove expired entries
            self._cleanup_expired()
            
            # Add new entry
            self.entries.append(entry)
            logger.debug(f"Added entry to short-term memory: {entry.id}")
    
    def add_conversation_turn(self, turn: ConversationTurn) -> None:
        """Add conversation turn to history."""
        with self.lock:
            self.conversation_history.append(turn)
            
            # Keep only recent conversation history
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            logger.debug(f"Added conversation turn: {turn.turn_id}")
    
    def get_recent_context(self, max_entries: int = 10) -> List[MemoryEntry]:
        """Get recent context entries."""
        with self.lock:
            self._cleanup_expired()
            
            # Sort by importance and recency
            sorted_entries = sorted(
                self.entries,
                key=lambda x: (x.importance, x.timestamp),
                reverse=True
            )
            
            return list(sorted_entries[:max_entries])
    
    def get_conversation_context(self, max_turns: int = 5) -> List[ConversationTurn]:
        """Get recent conversation context."""
        with self.lock:
            return list(self.conversation_history[-max_turns:])
    
    def update_context(self, key: str, value: Any) -> None:
        """Update current context."""
        with self.lock:
            self.current_context[key] = value
            logger.debug(f"Updated context: {key}")
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context value."""
        with self.lock:
            return self.current_context.get(key, default)
    
    def clear_context(self) -> None:
        """Clear current context."""
        with self.lock:
            self.current_context.clear()
            logger.debug("Cleared short-term context")
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        now = datetime.now()
        
        # Remove expired entries
        while self.entries and (now - self.entries[0].timestamp) > self.ttl:
            expired_entry = self.entries.popleft()
            logger.debug(f"Removed expired entry: {expired_entry.id}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self.lock:
            self._cleanup_expired()
            
            return {
                'total_entries': len(self.entries),
                'conversation_turns': len(self.conversation_history),
                'context_keys': len(self.current_context),
                'memory_usage': f"{len(self.entries)}/{self.max_size}",
                'oldest_entry': self.entries[0].timestamp if self.entries else None,
                'newest_entry': self.entries[-1].timestamp if self.entries else None
            }


class LongTermMemory:
    """Manages long-term memory for persistent knowledge."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize long-term memory.
        
        Args:
            storage_path: Path to store memory files
        """
        self.storage_path = storage_path or settings.data_dir / "memory"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.memories: Dict[str, MemoryEntry] = {}
        self.importance_threshold = 0.7  # Threshold for long-term storage
        self.lock = threading.Lock()
        
        # Load existing memories
        self._load_memories()
        
        logger.info(f"Long-term memory initialized: {len(self.memories)} entries loaded")
    
    def add_entry(self, entry: MemoryEntry) -> None:
        """Add entry to long-term memory."""
        with self.lock:
            self.memories[entry.id] = entry
            
            # Save to disk if important enough
            if entry.importance >= self.importance_threshold:
                self._save_entry(entry)
            
            logger.debug(f"Added entry to long-term memory: {entry.id}")
    
    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get specific memory entry."""
        with self.lock:
            entry = self.memories.get(entry_id)
            if entry:
                entry.access_count += 1
                entry.last_accessed = datetime.now()
            return entry
    
    def search_memories(self, query: str, memory_type: Optional[str] = None,
                       min_importance: float = 0.0, limit: int = 10) -> List[MemoryEntry]:
        """Search memories by content and criteria."""
        with self.lock:
            results = []
            query_lower = query.lower()
            
            for entry in self.memories.values():
                # Filter by type if specified
                if memory_type and entry.memory_type != memory_type:
                    continue
                
                # Filter by importance
                if entry.importance < min_importance:
                    continue
                
                # Simple text search (could be enhanced with embeddings)
                if query_lower in entry.content.lower():
                    entry.access_count += 1
                    entry.last_accessed = datetime.now()
                    results.append(entry)
            
            # Sort by importance and access frequency
            results.sort(
                key=lambda x: (x.importance, x.access_count, x.timestamp),
                reverse=True
            )
            
            return results[:limit]
    
    def consolidate_from_short_term(self, short_term_entries: List[MemoryEntry]) -> int:
        """Consolidate important entries from short-term memory."""
        consolidated = 0
        
        for entry in short_term_entries:
            if entry.importance >= self.importance_threshold:
                # Check if already exists
                if entry.id not in self.memories:
                    self.add_entry(entry)
                    consolidated += 1
                else:
                    # Update existing entry
                    existing = self.memories[entry.id]
                    existing.access_count += 1
                    existing.last_accessed = datetime.now()
                    # Increase importance based on repeated access
                    existing.importance = min(1.0, existing.importance + 0.1)
        
        logger.info(f"Consolidated {consolidated} entries from short-term memory")
        return consolidated
    
    def _save_entry(self, entry: MemoryEntry) -> None:
        """Save entry to disk."""
        try:
            file_path = self.storage_path / f"{entry.id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                # Convert datetime objects to strings for JSON serialization
                entry_dict = asdict(entry)
                entry_dict['timestamp'] = entry.timestamp.isoformat()
                if entry.last_accessed:
                    entry_dict['last_accessed'] = entry.last_accessed.isoformat()
                
                json.dump(entry_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save memory entry {entry.id}: {e}")
    
    def _load_memories(self) -> None:
        """Load memories from disk."""
        try:
            for file_path in self.storage_path.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        entry_dict = json.load(f)
                    
                    # Convert string timestamps back to datetime
                    entry_dict['timestamp'] = datetime.fromisoformat(entry_dict['timestamp'])
                    if entry_dict.get('last_accessed'):
                        entry_dict['last_accessed'] = datetime.fromisoformat(entry_dict['last_accessed'])
                    
                    entry = MemoryEntry(**entry_dict)
                    self.memories[entry.id] = entry
                    
                except Exception as e:
                    logger.warning(f"Failed to load memory file {file_path}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
    
    def cleanup_old_memories(self, days_threshold: int = 30) -> int:
        """Clean up old, rarely accessed memories."""
        with self.lock:
            cutoff_date = datetime.now() - timedelta(days=days_threshold)
            to_remove = []
            
            for entry_id, entry in self.memories.items():
                # Remove if old and rarely accessed
                last_access = entry.last_accessed or entry.timestamp
                if (last_access < cutoff_date and 
                    entry.access_count < 5 and 
                    entry.importance < 0.8):
                    to_remove.append(entry_id)
            
            # Remove entries
            for entry_id in to_remove:
                del self.memories[entry_id]
                # Remove file
                file_path = self.storage_path / f"{entry_id}.json"
                if file_path.exists():
                    file_path.unlink()
            
            logger.info(f"Cleaned up {len(to_remove)} old memories")
            return len(to_remove)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self.lock:
            if not self.memories:
                return {'total_entries': 0}
            
            importance_scores = [entry.importance for entry in self.memories.values()]
            access_counts = [entry.access_count for entry in self.memories.values()]
            
            return {
                'total_entries': len(self.memories),
                'avg_importance': sum(importance_scores) / len(importance_scores),
                'max_importance': max(importance_scores),
                'avg_access_count': sum(access_counts) / len(access_counts),
                'max_access_count': max(access_counts),
                'storage_path': str(self.storage_path),
                'disk_files': len(list(self.storage_path.glob("*.json")))
            }


class MemoryManager:
    """Manages both short-term and long-term memory systems."""
    
    def __init__(self):
        """Initialize memory manager."""
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()
        self.consolidation_interval = 300  # 5 minutes
        self.last_consolidation = time.time()
        
        logger.info("Memory manager initialized")
    
    def add_conversation_turn(self, user_input: str, assistant_response: str,
                            context_used: List[str], processing_time: float,
                            confidence_score: float) -> str:
        """Add a conversation turn to memory."""
        turn_id = f"turn_{int(time.time() * 1000)}"
        
        turn = ConversationTurn(
            turn_id=turn_id,
            user_input=user_input,
            assistant_response=assistant_response,
            context_used=context_used,
            timestamp=datetime.now(),
            processing_time=processing_time,
            confidence_score=confidence_score
        )
        
        self.short_term.add_conversation_turn(turn)
        
        # Create memory entry for important conversations
        if confidence_score > 0.8 or len(assistant_response) > 500:
            memory_entry = MemoryEntry(
                id=f"conv_{turn_id}",
                content=f"Q: {user_input}\nA: {assistant_response}",
                memory_type="conversation",
                importance=confidence_score,
                timestamp=datetime.now(),
                source="conversation",
                metadata={
                    'processing_time': processing_time,
                    'context_used': context_used
                }
            )
            
            self.short_term.add_entry(memory_entry)
        
        # Periodic consolidation
        self._maybe_consolidate()
        
        return turn_id
    
    def add_fact(self, content: str, source: str, importance: float = 0.8,
                metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a factual memory entry."""
        entry_id = f"fact_{int(time.time() * 1000)}"
        
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            memory_type="fact",
            importance=importance,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata or {}
        )
        
        if importance >= 0.7:
            self.long_term.add_entry(entry)
        else:
            self.short_term.add_entry(entry)
        
        return entry_id
    
    def get_relevant_context(self, query: str, max_entries: int = 5) -> List[MemoryEntry]:
        """Get relevant context for a query."""
        # Get from both short-term and long-term memory
        short_term_context = self.short_term.get_recent_context(max_entries // 2)
        long_term_context = self.long_term.search_memories(query, limit=max_entries // 2)
        
        # Combine and sort by relevance
        all_context = short_term_context + long_term_context
        all_context.sort(key=lambda x: x.importance, reverse=True)
        
        return all_context[:max_entries]
    
    def get_conversation_context(self, max_turns: int = 3) -> List[ConversationTurn]:
        """Get recent conversation context."""
        return self.short_term.get_conversation_context(max_turns)
    
    def _maybe_consolidate(self) -> None:
        """Consolidate memories if enough time has passed."""
        current_time = time.time()
        if current_time - self.last_consolidation > self.consolidation_interval:
            self._consolidate_memories()
            self.last_consolidation = current_time
    
    def _consolidate_memories(self) -> None:
        """Consolidate important short-term memories to long-term."""
        short_term_entries = self.short_term.get_recent_context(50)
        consolidated = self.long_term.consolidate_from_short_term(short_term_entries)
        
        if consolidated > 0:
            logger.info(f"Memory consolidation completed: {consolidated} entries moved to long-term")
    
    def cleanup_memories(self) -> Dict[str, int]:
        """Clean up old memories."""
        long_term_cleaned = self.long_term.cleanup_old_memories()
        
        return {
            'long_term_cleaned': long_term_cleaned
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        return {
            'short_term': self.short_term.get_memory_stats(),
            'long_term': self.long_term.get_memory_stats(),
            'last_consolidation': datetime.fromtimestamp(self.last_consolidation).isoformat()
        }


# Global memory manager instance
memory_manager = MemoryManager()
