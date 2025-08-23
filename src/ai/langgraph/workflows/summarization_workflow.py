from typing import TypedDict, List, Dict, Any

try:
    from langgraph.graph import StateGraph, END
except Exception:  # pragma: no cover
    StateGraph = None  # type: ignore
    END = "__END__"  # type: ignore


class SummarizationState(TypedDict, total=False):
    content: str
    summary: str
    chunks: List[str]
    confidence: float
    error: str
    max_retries: int


def _chunk_node(state: SummarizationState) -> SummarizationState:
    content = state.get("content", "")
    if not content or not content.strip():
        return {"error": "missing content", "chunks": []}
    
    # naive chunking fallback (LangChain splitter can replace later)
    chunk_size = 1500
    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
    
    # Filter out empty chunks
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    if not chunks:
        return {"error": "no valid chunks created", "chunks": []}
    
    return {"chunks": chunks}


def _summary_node(state: SummarizationState) -> SummarizationState:
    llm_client = state.get("llm_client")
    prompt_manager = state.get("prompt_manager")
    chunks = state.get("chunks", [])
    original_content = state.get("content", "")
    
    # Use chunks if available, otherwise fallback to original content
    if chunks:
        content = "\n\n".join(chunks)
    else:
        content = original_content
    
    addl = state.get("additional_instructions", "")
    gen_max_tokens = int(state.get("max_tokens", 0)) or 1600
    
    if not llm_client or not prompt_manager:
        return {"error": "missing llm_client/prompt_manager"}
    
    if not content or not content.strip():
        return {"error": "missing content", "confidence": 0.0}
    
    try:
        messages = prompt_manager.build_summary_prompt(
            content=content,
            additional_instructions=addl,
        )
        resp = llm_client.generate_response(messages, max_tokens=gen_max_tokens, temperature=0.2)
        
        # Extract content safely
        summary_content = resp.get("content", "") if isinstance(resp, dict) else getattr(resp, "content", "")
        
        # Calculate dynamic confidence based on response quality
        if summary_content:
            # Simple confidence calculation based on summary length vs content length
            content_words = len(content.split())
            summary_words = len(summary_content.split())
            if content_words > 0:
                ratio = summary_words / content_words
                # Prefer 10%-35% ratio for good summaries
                if 0.1 <= ratio <= 0.35:
                    confidence = 0.9
                elif ratio < 0.1:
                    confidence = 0.7
                else:
                    confidence = 0.75
            else:
                confidence = 0.8
        else:
            confidence = 0.5
            
        return {"summary": summary_content, "confidence": confidence}
        
    except Exception as e:
        return {"error": f"summary generation failed: {str(e)}", "confidence": 0.3}


def _validate_node(state: SummarizationState) -> SummarizationState:
    # Get content from either chunks or original content
    chunks = state.get("chunks", [])
    original_content = state.get("content", "")
    summary = state.get("summary", "")
    
    # Determine which content to use for validation
    if chunks:
        content = "\n\n".join(chunks)
    else:
        content = original_content
    
    if not content or not content.strip():
        retries = state.get("max_retries", 0) + 1
        return {"error": "missing content", "confidence": 0.0, "max_retries": retries}
    
    if not summary or not summary.strip():
        retries = state.get("max_retries", 0) + 1
        return {"error": "missing summary", "confidence": 0.0, "max_retries": retries}
    
    try:
        cw = max(1, len(content.split()))
        sw = len(summary.split())
        ratio = sw / cw
        
        # Enhanced confidence calculation
        if 0.1 <= ratio <= 0.35:
            # Optimal summary length
            confidence = 0.9
        elif ratio < 0.1:
            # Too short summary
            confidence = 0.6
        elif ratio <= 0.5:
            # Acceptable but long
            confidence = 0.7
        else:
            # Too long summary
            confidence = 0.5
            
        # Additional quality checks
        if summary.strip() and len(summary) > 50:
            confidence = min(confidence + 0.1, 1.0)
        
        retries = state.get("max_retries", 0) + 1
        return {"confidence": confidence, "max_retries": retries}
        
    except Exception as e:
        return {"error": f"validation failed: {str(e)}", "confidence": 0.3}


def _should_continue(state: SummarizationState) -> str:
    confidence = state.get("confidence", 0.0)
    retries = state.get("max_retries", 0)
    error = state.get("error", "")
    
    # Always end if there's an error to prevent infinite loops
    if error:
        return "end"
    
    # Always end after max retries to prevent infinite loops
    if retries >= 2:
        return "end"
    
    if int(state.get("remaining_loops", 0)) <= 0:
        return "end"
    
    # End if confidence is high enough
    if confidence >= 0.85:
        return "end"
    
    return "summarize"


def create_summarization_workflow(recursion_limit: int = 10):
    if StateGraph is None:  # fallback when LangGraph not installed
        def _fallback_invoke(initial_state: SummarizationState) -> SummarizationState:  # type: ignore[misc]
            state = dict(initial_state)
            state.update(_chunk_node(state))
            state.update(_summary_node(state))
            state.update(_validate_node(state))
            return state

        class _Fallback:
            def invoke(self, initial_state: SummarizationState) -> SummarizationState:
                return _fallback_invoke(initial_state)

        return _Fallback()

    graph = StateGraph(SummarizationState)
    graph.add_node("chunk", _chunk_node)
    graph.add_node("summarize", _summary_node)
    graph.add_node("validate", _validate_node)
    graph.set_entry_point("chunk")
    graph.add_edge("chunk", "summarize")
    graph.add_edge("summarize", "validate")
    graph.add_conditional_edges("validate", _should_continue, {"summarize": "summarize", "end": END})
    
    # Compile graph
    _compiled = graph.compile()

    # Return a thin wrapper that injects recursion_limit on each invoke
    class _WithConfig:
        def __init__(self, g, limit: int):
            self._g = g
            self._limit = limit

        def invoke(self, initial_state: SummarizationState) -> SummarizationState:  # type: ignore[misc]
            try:
                return self._g.invoke(initial_state, config={"recursion_limit": self._limit})
            except TypeError:
                # Older versions may not accept config; fallback to plain invoke
                return self._g.invoke(initial_state)

    return _WithConfig(_compiled, recursion_limit)


