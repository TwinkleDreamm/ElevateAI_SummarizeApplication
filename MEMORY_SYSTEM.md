# ElevateAI Memory System Documentation

## 🧠 Overview

ElevateAI implements a sophisticated dual-memory system inspired by human cognitive architecture, featuring both short-term and long-term memory components that work together to provide contextual awareness and conversation continuity.

## 🏗️ Architecture

### Memory Components

```
┌─────────────────┐    ┌──────────────────┐
│  Short-term     │    │   Long-term      │
│  Memory         │◄──►│   Memory         │
│                 │    │                  │
│ • Conversations │    │ • Important Facts│
│ • Context       │    │ • Learned Info   │
│ • Temp Data     │    │ • User Patterns  │
│ • Session Info  │    │ • Summaries      │
└─────────────────┘    └──────────────────┘
         │                       │
         └───────────┬───────────┘
                     │
            ┌─────────────────┐
            │ Memory Manager  │
            │                 │
            │ • Consolidation │
            │ • Context Mgmt  │
            │ • Search        │
            │ • Integration   │
            └─────────────────┘
```

## 📊 Memory Types

### Short-term Memory
- **Lifespan**: 30 minutes (configurable)
- **Capacity**: 50 entries (LRU eviction)
- **Storage**: In-memory only
- **Purpose**: Active conversation context

**Content Types:**
- Conversation turns (user input + assistant response)
- Current session context
- Temporary observations
- Processing metadata

### Long-term Memory
- **Lifespan**: Persistent (until manually cleaned)
- **Capacity**: Unlimited (disk-based)
- **Storage**: JSON files + in-memory index
- **Purpose**: Persistent knowledge and patterns

**Content Types:**
- Important facts and information
- High-confidence conversation excerpts
- User preferences and patterns
- Document insights and summaries

## 🔄 Memory Consolidation

### Automatic Consolidation
- **Frequency**: Every 5 minutes
- **Triggers**: 
  - Importance threshold ≥ 0.7
  - High confidence scores ≥ 0.8
  - Frequent access patterns
  - Long conversation responses

### Consolidation Process
1. **Evaluation**: Assess importance and relevance
2. **Filtering**: Apply thresholds and criteria
3. **Transfer**: Move qualifying entries to long-term
4. **Indexing**: Update search indices
5. **Cleanup**: Remove expired short-term entries

## 🎯 Memory Integration

### Context Enhancement
```python
# Example: Memory-enhanced prompt
messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant. Use the following context from previous interactions when relevant:\n\nRecent conversation:\nUser: What is machine learning?\nAssistant: Machine learning is a subset of AI...\n\nRelevant context:\n- Previous discussion about neural networks\n- User interested in practical applications"
    },
    {
        "role": "user", 
        "content": "How does deep learning relate to what we discussed?"
    }
]
```

### Memory Search
- **Semantic Search**: Find relevant memories by content similarity
- **Temporal Search**: Retrieve recent or historical context
- **Type-based Search**: Filter by memory type (fact, conversation, etc.)
- **Importance-based**: Prioritize high-importance memories

## 🛠️ Configuration

### Memory Settings
```python
# Short-term memory configuration
SHORT_TERM_CONFIG = {
    'max_size': 50,           # Maximum entries
    'ttl_minutes': 30,        # Time to live
    'cleanup_interval': 300   # Cleanup frequency (seconds)
}

# Long-term memory configuration
LONG_TERM_CONFIG = {
    'importance_threshold': 0.7,  # Consolidation threshold
    'storage_path': 'data/memory', # Storage directory
    'max_search_results': 10      # Search result limit
}

# Memory manager configuration
MEMORY_MANAGER_CONFIG = {
    'consolidation_interval': 300,  # 5 minutes
    'enable_auto_consolidation': True,
    'max_context_entries': 5
}
```

### Environment Variables
```env
# Memory system settings
MEMORY_ENABLED=true
MEMORY_STORAGE_PATH=data/memory
MEMORY_CONSOLIDATION_INTERVAL=300
MEMORY_IMPORTANCE_THRESHOLD=0.7
```

## 📈 Usage Examples

### Basic Memory Operations
```python
from src.utils.memory import memory_manager

# Add a conversation turn
turn_id = memory_manager.add_conversation_turn(
    user_input="What is quantum computing?",
    assistant_response="Quantum computing uses quantum mechanics...",
    context_used=["physics_context"],
    processing_time=2.5,
    confidence_score=0.9
)

# Add a fact
fact_id = memory_manager.add_fact(
    content="Python was created by Guido van Rossum in 1991",
    source="programming_history",
    importance=0.8,
    metadata={"category": "programming", "language": "python"}
)

# Get relevant context for a query
context = memory_manager.get_relevant_context(
    query="Tell me about Python programming",
    max_entries=5
)

# Get conversation history
history = memory_manager.get_conversation_context(max_turns=3)
```

### Memory-Enhanced LLM Calls
```python
from src.ai.llm_client import LLMClient

llm = LLMClient()

# Generate response with memory integration
response = llm.generate_response(
    messages=[
        {"role": "user", "content": "Continue our discussion about AI"}
    ],
    use_memory=True,              # Enable memory integration
    store_in_memory=True,         # Store this conversation
    max_memory_context=3          # Max memory entries to use
)
```

## 📊 Memory Statistics

### Monitoring Memory Usage
```python
# Get comprehensive memory statistics
stats = memory_manager.get_memory_stats()

print(f"Short-term entries: {stats['short_term']['total_entries']}")
print(f"Long-term entries: {stats['long_term']['total_entries']}")
print(f"Average importance: {stats['long_term']['avg_importance']:.2f}")
```

### Memory Cleanup
```python
# Manual cleanup of old memories
cleaned = memory_manager.cleanup_memories()
print(f"Cleaned {cleaned['long_term_cleaned']} old memories")

# Clear short-term memory
memory_manager.short_term.clear_context()
```

## 🎨 Web Interface Integration

### Memory Controls in Streamlit
- **Memory Statistics**: View current memory usage and statistics
- **Conversation History**: Browse recent conversation turns
- **Memory Controls**: Clear, consolidate, or cleanup memories
- **Memory Settings**: Configure memory behavior

### Memory Visualization
- Memory usage graphs
- Importance distribution charts
- Access pattern analysis
- Consolidation timeline

## 🔧 Advanced Features

### Custom Memory Types
```python
# Define custom memory entry
custom_entry = MemoryEntry(
    id="custom_1",
    content="User prefers technical explanations",
    memory_type="preference",
    importance=0.9,
    timestamp=datetime.now(),
    source="user_interaction",
    metadata={"category": "preference", "user_id": "user123"}
)
```

### Memory Search Filters
```python
# Search with advanced filters
results = long_term_memory.search_memories(
    query="machine learning",
    memory_type="fact",
    min_importance=0.8,
    limit=10
)
```

### Batch Memory Operations
```python
# Batch consolidation
entries_to_consolidate = short_term_memory.get_recent_context(20)
consolidated_count = long_term_memory.consolidate_from_short_term(entries_to_consolidate)
```

## 🚀 Performance Considerations

### Memory Optimization
- **Lazy Loading**: Load memories on-demand
- **Indexing**: Fast search with in-memory indices
- **Compression**: Efficient storage of large memories
- **Caching**: Cache frequently accessed memories

### Scalability
- **Distributed Memory**: Support for shared memory across instances
- **Memory Sharding**: Partition memories by user or topic
- **Async Operations**: Non-blocking memory operations
- **Background Processing**: Consolidation in background threads

## 🧪 Testing

### Memory System Tests
```bash
# Run memory system tests
pytest tests/test_memory_system.py -v

# Test specific memory components
pytest tests/test_memory_system.py::TestShortTermMemory -v
pytest tests/test_memory_system.py::TestLongTermMemory -v
pytest tests/test_memory_system.py::TestMemoryManager -v
```

### Memory Integration Tests
```bash
# Test memory integration with LLM
pytest tests/test_llm_memory_integration.py -v

# Test memory in web interface
pytest tests/test_streamlit_memory.py -v
```

## 🔮 Future Enhancements

### Planned Features
- **Semantic Memory Search**: Vector-based memory retrieval
- **Memory Clustering**: Group related memories automatically
- **User-specific Memory**: Separate memory spaces per user
- **Memory Export/Import**: Backup and restore memory data
- **Memory Analytics**: Advanced memory usage analytics

### Advanced Memory Types
- **Episodic Memory**: Detailed conversation episodes
- **Procedural Memory**: Learned processes and workflows
- **Semantic Networks**: Connected knowledge graphs
- **Emotional Memory**: Sentiment and emotional context

This memory system provides ElevateAI with human-like conversation continuity and contextual awareness, making interactions more natural and personalized over time.
