from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from src.search.semantic_search import SemanticSearchEngine

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    k: int = 10
    threshold: float = 0.0
    filters: Optional[Dict[str, Any]] = None

class SearchResultItem(BaseModel):
    id: int
    score: float
    text: str
    metadata: Dict[str, Any]
    rank: int

class SearchResponse(BaseModel):
    results: List[SearchResultItem]

_engine = None

def _get_engine():
    global _engine
    if _engine is None:
        _engine = SemanticSearchEngine()
    return _engine

@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    engine = _get_engine()
    raw = engine.search(req.query, k=req.k, threshold=req.threshold)
    # Convert to response items
    resp_items: List[SearchResultItem] = []
    for i, r in enumerate(raw):
        resp_items.append(SearchResultItem(
            id=r['id'],
            score=r['score'],
            text=r['metadata'].get('text', ''),
            metadata=r['metadata'],
            rank=i+1
        ))
    return {"results": resp_items}
