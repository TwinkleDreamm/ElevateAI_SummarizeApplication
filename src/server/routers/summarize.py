from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from src.ai.summarizer import Summarizer
from src.search.semantic_search import SemanticSearchEngine

router = APIRouter()

class Chunk(BaseModel):
    text: str
    metadata: Dict[str, Any] = {}

class SummarizeRequest(BaseModel):
    chunks: List[Chunk]
    query: Optional[str] = None
    use_chain_of_thought: bool = False
    max_tokens: int = 800

class SummarizeResponse(BaseModel):
    summary: str
    summary_type: Optional[str] = None
    word_count: Optional[int] = None
    compression_ratio: Optional[float] = None
    source_info: Optional[Dict[str, Any]] = None

_summarizer = None

def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = Summarizer()
    return _summarizer

_engine = None

def _get_engine():
    global _engine
    if _engine is None:
        _engine = SemanticSearchEngine()
    return _engine

@router.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    sm = _get_summarizer()
    # Convert chunks to expected internal format
    chunk_dicts = [{"text": c.text, **c.metadata} for c in req.chunks]
    result = sm.summarize_chunks(
        chunk_dicts,
        query=req.query,
        use_chain_of_thought=req.use_chain_of_thought,
        max_tokens=req.max_tokens,
    )
    return {
        "summary": result.summary,
        "summary_type": getattr(result.summary_type, "value", None) if hasattr(result, "summary_type") else None,
        "word_count": getattr(result, "word_count", None),
        "compression_ratio": getattr(result, "compression_ratio", None),
        "source_info": getattr(result, "source_info", None),
    }

class SummarizeFromSearchRequest(BaseModel):
    query: str
    k: int = 8
    threshold: float = 0.0

@router.post("/summarize-from-search", response_model=SummarizeResponse)
def summarize_from_search(req: SummarizeFromSearchRequest):
    engine = _get_engine()
    sm = _get_summarizer()
    raw = engine.search(req.query, k=req.k, threshold=req.threshold)
    chunks = [{"text": r["metadata"].get("text", "")} for r in raw]
    if not chunks:
        return {"summary": "", "summary_type": None, "word_count": 0, "compression_ratio": 0.0, "source_info": {}}
    res = sm.summarize_chunks(chunks, query=req.query)
    return {
        "summary": res.summary,
        "summary_type": getattr(res.summary_type, "value", None) if hasattr(res, "summary_type") else None,
        "word_count": getattr(res, "word_count", None),
        "compression_ratio": getattr(res, "compression_ratio", None),
        "source_info": getattr(res, "source_info", None),
    }
