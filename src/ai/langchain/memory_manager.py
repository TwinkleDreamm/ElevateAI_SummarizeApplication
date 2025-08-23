from typing import List, Dict, Any, Optional

try:
    from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
    from langchain.schema import HumanMessage, AIMessage
except Exception:  # pragma: no cover
    ConversationBufferWindowMemory = None  # type: ignore
    ConversationSummaryMemory = None  # type: ignore
    HumanMessage = None  # type: ignore
    AIMessage = None  # type: ignore


class LangchainMemoryManager:
    """Conversation memory using LangChain (optional dependency)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, llm: Optional[Any] = None) -> None:
        self.config = config or {}
        # Accept an external LLM (e.g., ChatOpenAI) for summary memory
        self.llm = llm
        self._init_memory()

    def _init_memory(self) -> None:
        window_size = int(self.config.get("memory_window_size", 5))
        if ConversationBufferWindowMemory is None:
            self.window_memory = None
            self.summary_memory = None
            return
        self.window_memory = ConversationBufferWindowMemory(
            k=window_size, return_messages=True, memory_key="chat_history"
        )
        # Only construct summary memory if we have a valid LLM instance
        if self.llm is not None:
            self.summary_memory = ConversationSummaryMemory(
                llm=self.llm,
                max_token_limit=int(self.config.get("memory_summary_threshold", 1000)),
                return_messages=True,
                memory_key="chat_summary",
            )
        else:
            self.summary_memory = None

    def add_message(self, role: str, content: str) -> None:
        if self.window_memory is None or HumanMessage is None:
            return
        if role in ("user", "human"):
            self.window_memory.chat_memory.add_message(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            self.window_memory.chat_memory.add_message(AIMessage(content=content))

    def get_recent_context(self, max_messages: int = 3) -> List[Dict[str, str]]:
        if self.window_memory is None:
            return []
        messages = self.window_memory.load_memory_variables({}).get("chat_history", [])
        last_msgs = messages[-max_messages:]
        result: List[Dict[str, str]] = []
        for m in last_msgs:
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            result.append({"role": role, "content": m.content})
        return result

    def clear(self) -> None:
        if self.window_memory:
            self.window_memory.clear()
        if self.summary_memory:
            self.summary_memory.clear()


