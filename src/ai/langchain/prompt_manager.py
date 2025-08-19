from typing import List, Dict, Any, Optional

try:
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema import HumanMessage, SystemMessage, AIMessage
except Exception:  # pragma: no cover
    ChatPromptTemplate = None  # type: ignore
    HumanMessage = None  # type: ignore
    SystemMessage = None  # type: ignore
    AIMessage = None  # type: ignore


class LangchainPromptManager:
    """Prompt management using LangChain templates (optional dependency)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._init_templates()

    def _init_templates(self) -> None:
        if ChatPromptTemplate is None:
            # Graceful no-op init when LangChain isn't installed
            self.qa_template = None
            self.summary_template = None
            self.analysis_template = None
            return

        self.qa_template = ChatPromptTemplate.from_messages([
            ("system", "You are a knowledgeable assistant that provides accurate, helpful answers based on the given context."),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
        ])

        self.summary_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert content summarizer. Create comprehensive, accurate, and well-structured summaries."),
            ("human", "Please summarize the following content:\n\n{content}\n\n{additional_instructions}\n\nSummary:")
        ])

        self.analysis_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert content analyst. Analyze the provided content and extract key insights."),
            ("human", "Analyze the following content:\n\n{content}\n\nAnalysis:")
        ])

    def build_qa_prompt(self, *, context: str, question: str, use_chain_of_thought: bool = False) -> List[Dict[str, str]]:
        if self.qa_template is None:
            # Fallback to legacy-compatible format
            if not context or context.strip() == "":
                # No context available
                messages = [
                    {"role": "system", "content": "You are a knowledgeable assistant. Answer the user's question directly and clearly."},
                    {"role": "user", "content": f"Question: {question}\n\nAnswer:"},
                ]
            else:
                # With context
                messages = [
                    {"role": "system", "content": "You are a knowledgeable assistant that provides accurate, helpful answers based on the given context."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
                ]
            
            if use_chain_of_thought:
                messages.insert(1, {"role": "system", "content": "Please think through this step by step before providing your final answer."})
            return messages

        # Handle empty context case for LangChain templates
        if not context or context.strip() == "":
            # Create a direct question prompt
            messages = [
                SystemMessage(content="You are a knowledgeable assistant. Answer the user's question directly and clearly."),
                HumanMessage(content=f"Question: {question}\n\nAnswer:")
            ]
        else:
            # Use normal template with context
            messages = self.qa_template.format_messages(context=context, question=question)
        
        if use_chain_of_thought:
            messages.insert(1, SystemMessage(content="Please think through this step by step before providing your final answer."))
        
        # Convert to dict for compatibility with existing llm_client
        return [{"role": msg.type, "content": msg.content} for msg in messages]

    def build_summary_prompt(self, *, content: str, additional_instructions: str = "") -> List[Dict[str, str]]:
        if self.summary_template is None:
            return [
                {"role": "system", "content": "You are an expert content summarizer. Create comprehensive, accurate, and well-structured summaries."},
                {"role": "user", "content": f"Please summarize the following content:\n\n{content}\n\n{additional_instructions}\n\nSummary:"},
            ]
        messages = self.summary_template.format_messages(content=content, additional_instructions=additional_instructions)
        return [{"role": msg.type, "content": msg.content} for msg in messages]

    def build_analysis_prompt(self, *, content: str) -> List[Dict[str, str]]:
        if self.analysis_template is None:
            return [
                {"role": "system", "content": "You are an expert content analyst. Analyze the provided content and extract key insights."},
                {"role": "user", "content": f"Analyze the following content:\n\n{content}\n\nAnalysis:"},
            ]
        messages = self.analysis_template.format_messages(content=content)
        return [{"role": msg.type, "content": msg.content} for msg in messages]


