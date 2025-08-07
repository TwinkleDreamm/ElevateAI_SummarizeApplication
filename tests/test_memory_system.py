"""
Tests for memory system functionality.
"""
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.memory import (
    MemoryEntry, ConversationTurn, ShortTermMemory, 
    LongTermMemory, MemoryManager, memory_manager
)


class TestMemoryEntry:
    """Test cases for MemoryEntry."""
    
    def test_memory_entry_creation(self):
        """Test creating a memory entry."""
        entry = MemoryEntry(
            id="test_1",
            content="This is a test memory",
            memory_type="fact",
            importance=0.8,
            timestamp=datetime.now(),
            source="test",
            metadata={"category": "test"}
        )
        
        assert entry.id == "test_1"
        assert entry.content == "This is a test memory"
        assert entry.memory_type == "fact"
        assert entry.importance == 0.8
        assert entry.access_count == 0


class TestShortTermMemory:
    """Test cases for ShortTermMemory."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.short_term = ShortTermMemory(max_size=10, ttl_minutes=1)
    
    def test_add_entry(self):
        """Test adding entries to short-term memory."""
        entry = MemoryEntry(
            id="test_1",
            content="Test content",
            memory_type="fact",
            importance=0.7,
            timestamp=datetime.now(),
            source="test",
            metadata={}
        )
        
        self.short_term.add_entry(entry)
        
        recent_context = self.short_term.get_recent_context(5)
        assert len(recent_context) == 1
        assert recent_context[0].id == "test_1"
    
    def test_conversation_history(self):
        """Test conversation history management."""
        turn = ConversationTurn(
            turn_id="turn_1",
            user_input="Hello",
            assistant_response="Hi there!",
            context_used=["context1"],
            timestamp=datetime.now(),
            processing_time=1.0,
            confidence_score=0.9
        )
        
        self.short_term.add_conversation_turn(turn)
        
        context = self.short_term.get_conversation_context(5)
        assert len(context) == 1
        assert context[0].turn_id == "turn_1"
        assert context[0].user_input == "Hello"
    
    def test_context_management(self):
        """Test context key-value management."""
        self.short_term.update_context("user_name", "Alice")
        self.short_term.update_context("topic", "AI")
        
        assert self.short_term.get_context("user_name") == "Alice"
        assert self.short_term.get_context("topic") == "AI"
        assert self.short_term.get_context("unknown", "default") == "default"
        
        self.short_term.clear_context()
        assert self.short_term.get_context("user_name") is None
    
    def test_memory_expiration(self):
        """Test memory entry expiration."""
        # Create entry with past timestamp
        old_entry = MemoryEntry(
            id="old_1",
            content="Old content",
            memory_type="fact",
            importance=0.5,
            timestamp=datetime.now() - timedelta(minutes=2),  # Expired
            source="test",
            metadata={}
        )
        
        new_entry = MemoryEntry(
            id="new_1",
            content="New content",
            memory_type="fact",
            importance=0.5,
            timestamp=datetime.now(),
            source="test",
            metadata={}
        )
        
        self.short_term.add_entry(old_entry)
        self.short_term.add_entry(new_entry)
        
        # Get recent context should only return non-expired entries
        recent_context = self.short_term.get_recent_context(10)
        assert len(recent_context) == 1
        assert recent_context[0].id == "new_1"


class TestLongTermMemory:
    """Test cases for LongTermMemory."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.long_term = LongTermMemory(storage_path=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_add_and_retrieve_entry(self):
        """Test adding and retrieving entries."""
        entry = MemoryEntry(
            id="long_1",
            content="Important fact",
            memory_type="fact",
            importance=0.9,
            timestamp=datetime.now(),
            source="test",
            metadata={"category": "important"}
        )
        
        self.long_term.add_entry(entry)
        
        retrieved = self.long_term.get_entry("long_1")
        assert retrieved is not None
        assert retrieved.content == "Important fact"
        assert retrieved.access_count == 1
    
    def test_search_memories(self):
        """Test searching memories."""
        entries = [
            MemoryEntry(
                id="search_1",
                content="Python programming language",
                memory_type="fact",
                importance=0.8,
                timestamp=datetime.now(),
                source="test",
                metadata={}
            ),
            MemoryEntry(
                id="search_2",
                content="JavaScript web development",
                memory_type="fact",
                importance=0.7,
                timestamp=datetime.now(),
                source="test",
                metadata={}
            ),
            MemoryEntry(
                id="search_3",
                content="Python data science",
                memory_type="fact",
                importance=0.9,
                timestamp=datetime.now(),
                source="test",
                metadata={}
            )
        ]
        
        for entry in entries:
            self.long_term.add_entry(entry)
        
        # Search for Python-related memories
        results = self.long_term.search_memories("Python", limit=5)
        assert len(results) == 2
        
        # Results should be sorted by importance
        assert results[0].importance >= results[1].importance
        
        # Search by memory type
        fact_results = self.long_term.search_memories("", memory_type="fact", limit=5)
        assert len(fact_results) == 3
    
    def test_consolidation(self):
        """Test consolidating from short-term memory."""
        short_term_entries = [
            MemoryEntry(
                id="consolidate_1",
                content="High importance fact",
                memory_type="fact",
                importance=0.9,  # Above threshold
                timestamp=datetime.now(),
                source="test",
                metadata={}
            ),
            MemoryEntry(
                id="consolidate_2",
                content="Low importance fact",
                memory_type="fact",
                importance=0.5,  # Below threshold
                timestamp=datetime.now(),
                source="test",
                metadata={}
            )
        ]
        
        consolidated = self.long_term.consolidate_from_short_term(short_term_entries)
        
        # Only high importance entry should be consolidated
        assert consolidated == 1
        assert self.long_term.get_entry("consolidate_1") is not None
        assert self.long_term.get_entry("consolidate_2") is None
    
    def test_persistence(self):
        """Test saving and loading memories."""
        entry = MemoryEntry(
            id="persist_1",
            content="Persistent memory",
            memory_type="fact",
            importance=0.8,
            timestamp=datetime.now(),
            source="test",
            metadata={"persistent": True}
        )
        
        self.long_term.add_entry(entry)
        
        # Create new instance and verify memory is loaded
        new_long_term = LongTermMemory(storage_path=self.temp_dir)
        retrieved = new_long_term.get_entry("persist_1")
        
        assert retrieved is not None
        assert retrieved.content == "Persistent memory"
        assert retrieved.metadata["persistent"] is True


class TestMemoryManager:
    """Test cases for MemoryManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.memory_manager = MemoryManager()
    
    def test_conversation_turn_management(self):
        """Test adding and retrieving conversation turns."""
        turn_id = self.memory_manager.add_conversation_turn(
            user_input="What is AI?",
            assistant_response="AI stands for Artificial Intelligence...",
            context_used=["context1", "context2"],
            processing_time=2.5,
            confidence_score=0.9
        )
        
        assert turn_id.startswith("turn_")
        
        # Get conversation context
        context = self.memory_manager.get_conversation_context(max_turns=5)
        assert len(context) == 1
        assert context[0].user_input == "What is AI?"
    
    def test_fact_management(self):
        """Test adding and retrieving facts."""
        fact_id = self.memory_manager.add_fact(
            content="The capital of France is Paris",
            source="geography_book",
            importance=0.8,
            metadata={"category": "geography"}
        )
        
        assert fact_id.startswith("fact_")
        
        # Search for relevant context
        context = self.memory_manager.get_relevant_context("France", max_entries=5)
        assert len(context) >= 1
        
        # Find our fact in the context
        found_fact = None
        for entry in context:
            if "Paris" in entry.content:
                found_fact = entry
                break
        
        assert found_fact is not None
        assert found_fact.memory_type == "fact"
    
    def test_memory_consolidation(self):
        """Test memory consolidation process."""
        # Add high-importance conversation
        self.memory_manager.add_conversation_turn(
            user_input="Important question about quantum physics",
            assistant_response="Quantum physics is a fundamental theory in physics...",
            context_used=[],
            processing_time=3.0,
            confidence_score=0.95  # High confidence
        )
        
        # Force consolidation
        self.memory_manager._consolidate_memories()
        
        # Check that important conversation was moved to long-term memory
        long_term_stats = self.memory_manager.long_term.get_memory_stats()
        assert long_term_stats['total_entries'] > 0
    
    def test_memory_statistics(self):
        """Test memory statistics retrieval."""
        # Add some data
        self.memory_manager.add_fact("Test fact", "test", importance=0.8)
        self.memory_manager.add_conversation_turn(
            "Test question", "Test answer", [], 1.0, 0.8
        )
        
        stats = self.memory_manager.get_memory_stats()
        
        assert 'short_term' in stats
        assert 'long_term' in stats
        assert 'last_consolidation' in stats
        
        short_stats = stats['short_term']
        assert short_stats['total_entries'] >= 0
        assert short_stats['conversation_turns'] >= 1


class TestMemoryIntegration:
    """Integration tests for memory system."""
    
    def test_global_memory_manager(self):
        """Test global memory manager instance."""
        # Test that global instance works
        turn_id = memory_manager.add_conversation_turn(
            "Global test", "Global response", [], 1.0, 0.8
        )
        
        assert turn_id is not None
        
        context = memory_manager.get_conversation_context(1)
        assert len(context) >= 1


if __name__ == "__main__":
    pytest.main([__file__])
