"""
Notebook store: CRUD for notebooks and their sources, persisted as JSON files.

Data model (stable, minimal):
- Notebook: { id, name, created_at, updated_at, is_favorite, description, tags, sources: [SourceRef], examples: [str] }
- SourceRef: { id, type, title, source_path_or_url, added_at, meta }

All notebooks saved under data/notebooks/notebooks.json
Each notebook can also have a folder data/notebooks/{id}/ for raw uploads, if needed.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from src.utils.logger import logger


NOTEBOOKS_ROOT = Path("data/notebooks")
NOTEBOOKS_FILE = NOTEBOOKS_ROOT / "notebooks.json"


def _ensure_dirs() -> None:
    NOTEBOOKS_ROOT.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


@dataclass
class SourceRef:
    id: str
    type: str  # file, url, youtube
    title: str
    source_path_or_url: str
    added_at: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Notebook:
    id: str
    name: str
    created_at: str
    updated_at: str
    is_favorite: bool = False
    description: str = ""
    tags: List[str] = field(default_factory=list)
    sources: List[SourceRef] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    overview: str = ""
    # Hash of sources used to generate overview/examples to avoid unnecessary regeneration
    generated_hash: str = ""


def _load_all() -> List[Notebook]:
    _ensure_dirs()
    if not NOTEBOOKS_FILE.exists():
        return []
    try:
        data = json.loads(NOTEBOOKS_FILE.read_text(encoding="utf-8"))
        notebooks: List[Notebook] = []
        for item in data:
            sources = [SourceRef(**s) for s in item.get("sources", [])]
            nb = Notebook(
                id=item["id"],
                name=item["name"],
                created_at=item["created_at"],
                updated_at=item.get("updated_at", item["created_at"]),
                is_favorite=item.get("is_favorite", False),
                description=item.get("description", ""),
                tags=item.get("tags", []),
                sources=sources,
                examples=item.get("examples", []),
                overview=item.get("overview", ""),
                generated_hash=item.get("generated_hash", ""),
            )
            notebooks.append(nb)
        return notebooks
    except Exception as e:
        logger.warning(f"Failed to load notebooks.json: {e}")
        return []


def _save_all(notebooks: List[Notebook]) -> None:
    _ensure_dirs()
    serializable = []
    for nb in notebooks:
        nb_dict = asdict(nb)
        nb_dict["sources"] = [asdict(s) for s in nb.sources]
        serializable.append(nb_dict)
    NOTEBOOKS_FILE.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def list_notebooks(query: str = "", favorite_only: bool = False, date_from: Optional[str] = None, date_to: Optional[str] = None) -> List[Notebook]:
    notebooks = _load_all()
    if query:
        q = query.lower()
        notebooks = [n for n in notebooks if q in n.name.lower() or q in n.description.lower() or any(q in t.lower() for t in n.tags)]
    if favorite_only:
        notebooks = [n for n in notebooks if n.is_favorite]
    if date_from:
        notebooks = [n for n in notebooks if n.created_at >= date_from]
    if date_to:
        notebooks = [n for n in notebooks if n.created_at <= date_to]
    # Sort by creation date first (most stable), then by update date
    notebooks.sort(key=lambda n: (n.created_at, n.updated_at or n.created_at), reverse=True)
    return notebooks


def get_notebook(notebook_id: str) -> Optional[Notebook]:
    for n in _load_all():
        if n.id == notebook_id:
            return n
    return None


def create_notebook(name: str, description: str = "", tags: Optional[List[str]] = None) -> Notebook:
    notebooks = _load_all()
    nb = Notebook(
        id=str(uuid.uuid4()),
        name=name,
        created_at=_now_iso(),
        updated_at=_now_iso(),
        is_favorite=False,
        description=description,
        tags=tags or [],
        sources=[],
        examples=[],
    )
    notebooks.append(nb)
    _save_all(notebooks)
    return nb


def rename_notebook(notebook_id: str, new_name: str) -> bool:
    notebooks = _load_all()
    changed = False
    for n in notebooks:
        if n.id == notebook_id:
            n.name = new_name
            n.updated_at = _now_iso()
            changed = True
            break
    if changed:
        _save_all(notebooks)
    return changed


def update_notebook(notebook_id: str, *, name: str, description: str = "", tags: Optional[List[str]] = None) -> bool:
    notebooks = _load_all()
    changed = False
    for n in notebooks:
        if n.id == notebook_id:
            n.name = name
            n.description = description
            n.tags = tags or []
            n.updated_at = _now_iso()
            changed = True
            break
    if changed:
        _save_all(notebooks)
    return changed


def toggle_favorite(notebook_id: str, value: Optional[bool] = None) -> bool:
    notebooks = _load_all()
    changed = False
    for n in notebooks:
        if n.id == notebook_id:
            n.is_favorite = (not n.is_favorite) if value is None else bool(value)
            n.updated_at = _now_iso()
            changed = True
            break
    if changed:
        _save_all(notebooks)
    return changed


def delete_notebook(notebook_id: str) -> bool:
    notebooks = _load_all()
    new_list = [n for n in notebooks if n.id != notebook_id]
    if len(new_list) != len(notebooks):
        _save_all(new_list)
        return True
    return False


def add_source(notebook_id: str, *, type: str, title: str, source_path_or_url: str, meta: Optional[Dict[str, Any]] = None) -> Optional[SourceRef]:
    notebooks = _load_all()
    for n in notebooks:
        if n.id == notebook_id:
            sref = SourceRef(
                id=str(uuid.uuid4()),
                type=type,
                title=title,
                source_path_or_url=source_path_or_url,
                added_at=_now_iso(),
                meta=meta or {},
            )
            n.sources.append(sref)
            n.updated_at = _now_iso()
            _save_all(notebooks)
            return sref
    return None


def remove_source(notebook_id: str, source_id: str) -> bool:
    notebooks = _load_all()
    changed = False
    for n in notebooks:
        if n.id == notebook_id:
            before = len(n.sources)
            n.sources = [s for s in n.sources if s.id != source_id]
            after = len(n.sources)
            if after != before:
                n.updated_at = _now_iso()
                changed = True
            break
    if changed:
        _save_all(notebooks)
    return changed


def update_examples(notebook_id: str, examples: List[str]) -> bool:
    notebooks = _load_all()
    for n in notebooks:
        if n.id == notebook_id:
            n.examples = examples[:5]
            n.updated_at = _now_iso()
            _save_all(notebooks)
            return True
    return False


def update_overview(notebook_id: str, overview: str) -> bool:
    """Update the overview for a notebook."""
    notebooks = _load_all()
    for n in notebooks:
        if n.id == notebook_id:
            n.overview = overview
            n.updated_at = _now_iso()
            _save_all(notebooks)
            return True
    return False


def get_overview(notebook_id: str) -> Optional[str]:
    """Get the overview for a notebook."""
    for n in _load_all():
        if n.id == notebook_id:
            return n.overview
    return None


def update_generated_hash(notebook_id: str, generated_hash: str) -> bool:
    """Persist the hash representing the sources used for the latest generation."""
    notebooks = _load_all()
    for n in notebooks:
        if n.id == notebook_id:
            n.generated_hash = generated_hash
            n.updated_at = _now_iso()
            _save_all(notebooks)
            return True
    return False


