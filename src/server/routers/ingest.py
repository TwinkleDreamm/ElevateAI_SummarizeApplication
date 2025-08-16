from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any

from src.database.vector_database import VectorDatabase

router = APIRouter()

class IngestTextItem(BaseModel):
    text: str
    metadata: Dict[str, Any] = {}

class IngestRequest(BaseModel):
    items: List[IngestTextItem]

class IngestResponse(BaseModel):
    ids: List[int]
    count: int

# Simple singleton
_vectordb = None

def _get_db():
    global _vectordb
    if _vectordb is None:
        _vectordb = VectorDatabase()
    return _vectordb

@router.post("/text", response_model=IngestResponse)
def ingest_text(req: IngestRequest):
    db = _get_db()
    texts = [i.text for i in req.items]
    metas = [i.metadata for i in req.items]
    ids = db.add_to_vectordb(texts, metas)
    return {"ids": ids, "count": len(ids)}
