"""
Vector database and embedding modules for ElevateAI.
"""

from .embedding_generator import EmbeddingGenerator
from .vector_database import VectorDatabase
from .metadata_manager import MetadataManager

__all__ = [
    "EmbeddingGenerator",
    "VectorDatabase",
    "MetadataManager"
]
