from typing import TypedDict, List, Dict, Any

try:
    from langgraph.graph import StateGraph, END
except Exception:  # pragma: no cover
    StateGraph = None  # type: ignore
    END = "__END__"  # type: ignore


class QAState(TypedDict, total=False):
    question: str
    search_results: List[Dict[str, Any]]
    context: str
    answer: str
    confidence: float
    sources: List[str]
    error: str
    max_retries: int


def _search_agent(state: QAState) -> QAState:
    question = state.get("question", "")
    search_engine = state.get("search_engine")
    
    if not question or search_engine is None:
        return {"error": "missing question or search_engine"}
    
    try:
        results = search_engine.search(question, k=8, threshold=0.2)
        
        if not results:
            # No search results found
            return {
                "search_results": [],
                "context": "",
                "error": "no_relevant_content_found"
            }
        
        formatted = [{"text": r.text or r.metadata.get("text", ""), "score": r.score, "metadata": r.metadata} for r in results]
        
        # Filter out empty or very short texts
        valid_texts = [r["text"] for r in formatted[:5] if r["text"] and len(r["text"].strip()) > 10]
        
        if not valid_texts:
            # No valid content found
            return {
                "search_results": formatted,
                "context": "",
                "error": "no_valid_content_found"
            }
        
        context = "\n\n".join(valid_texts)
        
        return {
            "search_results": formatted,
            "context": context,
            "search_success": True
        }
        
    except Exception as e:
        return {
            "error": f"search failed: {str(e)}",
            "search_results": [],
            "context": ""
        }


def _synthesis_agent(state: QAState) -> QAState:
    llm_client = state.get("llm_client")
    prompt_mgr = state.get("prompt_manager")
    context = state.get("context", "")
    question = state.get("question", "")
    
    if not llm_client or not prompt_mgr or not question:
        return {"error": "missing llm_client/prompt_manager/question"}
    
    # Handle empty context case
    if not context or context.strip() == "":
        # If no context, create a more direct prompt
        messages = [
            {"role": "system", "content": "You are a knowledgeable assistant. Answer the user's question directly and clearly."},
            {"role": "user", "content": f"Question: {question}\n\nAnswer:"}
        ]
    else:
        # Use normal QA prompt with context
        messages = prompt_mgr.build_qa_prompt(context=context, question=question)
    
    try:
        # Reduce max_tokens slightly when remaining loop budget is small to cut latency
        per_call = {"max_tokens": 600 if int(state.get("remaining_loops", 2)) <= 1 else None}
        resp = llm_client.generate_response(messages, **per_call)
        answer_content = resp.get("content", "") if isinstance(resp, dict) else getattr(resp, "content", "")
        
        # Calculate confidence based on answer quality
        if answer_content and len(answer_content.strip()) > 10:
            confidence = 0.8
        else:
            confidence = 0.5
            
        return {"answer": answer_content, "confidence": confidence}
        
    except Exception as e:
        return {"error": f"synthesis failed: {str(e)}", "confidence": 0.3}


def _validate_agent(state: QAState) -> QAState:
    answer = state.get("answer", "")
    question = state.get("question", "")
    if not answer or not question:
        # Increment retries even on failure to avoid infinite loops
        retries = state.get("max_retries", 0) + 1
        return {"error": "missing answer or question", "max_retries": retries}
    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())
    overlap = len(q_words.intersection(a_words)) / (len(q_words) or 1)
    confidence = (state.get("confidence", 0.8) + overlap) / 2
    # Increment retry counter here (stateful node), not in conditional fn
    retries = state.get("max_retries", 0) + 1
    # Decrement remaining loop budget (belt-and-suspenders guard)
    remaining = state.get("remaining_loops")
    if remaining is None:
        remaining = 2
    else:
        remaining = max(0, int(remaining) - 1)
    return {"confidence": confidence, "max_retries": retries, "remaining_loops": remaining}


def _should_continue(state: QAState) -> str:
    confidence = state.get("confidence", 0.0)
    retries = state.get("max_retries", 0)
    error = state.get("error", "")
    
    # Always end if there's an error to prevent infinite loops
    if error:
        return "end"
    
    # Hard stop conditions to prevent infinite loops
    if retries >= 2:
        return "end"
    if int(state.get("remaining_loops", 0)) <= 0:
        return "end"
    
    # End if confidence is high enough
    if confidence >= 0.8:
        return "end"
    
    return "synthesize"


def create_qa_workflow(recursion_limit: int = 10):
    if StateGraph is None:  # LangGraph not installed
        def _fallback_invoke(initial_state: QAState) -> QAState:  # type: ignore[misc]
            # Minimal fallback: run agents sequentially
            state = dict(initial_state)
            state.update(_search_agent(state))
            state.update(_synthesis_agent(state))
            state.update(_validate_agent(state))
            return state

        class _Fallback:
            def invoke(self, initial_state: QAState) -> QAState:
                return _fallback_invoke(initial_state)

        return _Fallback()

    graph = StateGraph(QAState)
    graph.add_node("search", _search_agent)
    graph.add_node("synthesize", _synthesis_agent)
    graph.add_node("validate", _validate_agent)
    graph.set_entry_point("search")
    graph.add_edge("search", "synthesize")
    graph.add_edge("synthesize", "validate")
    graph.add_conditional_edges("validate", _should_continue, {"synthesize": "synthesize", "end": END})
    
    # Compile graph
    _compiled = graph.compile()

    # Return a thin wrapper that injects recursion_limit on each invoke
    class _WithConfig:
        def __init__(self, g, limit: int):
            self._g = g
            self._limit = limit

        def invoke(self, initial_state: QAState) -> QAState:  # type: ignore[misc]
            try:
                return self._g.invoke(initial_state, config={"recursion_limit": self._limit})
            except TypeError:
                # Older versions may not accept config; fallback to plain invoke
                return self._g.invoke(initial_state)

    return _WithConfig(_compiled, recursion_limit)


