"""
Metadata management module for ElevateAI vector database.
Handles storage and retrieval of document metadata.
"""
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime

from config.settings import settings
from src.utils.logger import logger
from src.utils.exceptions import VectorDatabaseError


class MetadataManager:
    """Manages metadata for vector database entries."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the metadata manager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.settings = settings
        self.logger = logger
        
        # Metadata storage
        self.metadata_store: Dict[int, Dict[str, Any]] = {}
        self.deleted_ids: set = set()
        
        # Statistics
        self.total_added = 0
        self.total_deleted = 0
    
    def add_metadata(self, vector_id: int, metadata: Dict[str, Any]) -> None:
        """
        Add metadata for a vector.
        
        Args:
            vector_id: Vector ID
            metadata: Metadata dictionary
        """
        try:
            # Enhance metadata with system information
            enhanced_metadata = {
                **metadata,
                'vector_id': vector_id,
                'added_at': metadata.get('added_at', datetime.now().isoformat()),
                'status': 'active'
            }
            
            self.metadata_store[vector_id] = enhanced_metadata
            self.total_added += 1
            
            # Remove from deleted set if it was previously deleted
            self.deleted_ids.discard(vector_id)
            
        except Exception as e:
            self.logger.error(f"Failed to add metadata for vector {vector_id}: {e}")
            raise VectorDatabaseError(f"Failed to add metadata: {e}")
    
    def get_metadata(self, vector_id: int) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a vector.
        
        Args:
            vector_id: Vector ID
            
        Returns:
            Metadata dictionary or None if not found
        """
        if vector_id in self.deleted_ids:
            return None
        
        return self.metadata_store.get(vector_id)
    
    def update_metadata(self, vector_id: int, updates: Dict[str, Any]) -> bool:
        """
        Update metadata for a vector.
        
        Args:
            vector_id: Vector ID
            updates: Dictionary of updates to apply
            
        Returns:
            True if update successful, False if vector not found
        """
        if vector_id in self.deleted_ids or vector_id not in self.metadata_store:
            return False
        
        try:
            self.metadata_store[vector_id].update(updates)
            self.metadata_store[vector_id]['updated_at'] = datetime.now().isoformat()
            return True
        except Exception as e:
            self.logger.error(f"Failed to update metadata for vector {vector_id}: {e}")
            return False
    
    def delete_metadata(self, vector_id: int) -> bool:
        """
        Mark metadata as deleted (soft delete).
        
        Args:
            vector_id: Vector ID
            
        Returns:
            True if deletion successful
        """
        if vector_id in self.metadata_store:
            self.deleted_ids.add(vector_id)
            self.metadata_store[vector_id]['status'] = 'deleted'
            self.metadata_store[vector_id]['deleted_at'] = datetime.now().isoformat()
            self.total_deleted += 1
            return True
        return False
    
    def search_metadata(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search metadata based on filters.
        
        Args:
            filters: Dictionary of filter criteria
            
        Returns:
            List of matching metadata entries
        """
        results = []
        
        for vector_id, metadata in self.metadata_store.items():
            if vector_id in self.deleted_ids:
                continue
            
            # Check if metadata matches all filters
            matches = True
            for key, value in filters.items():
                if key not in metadata:
                    matches = False
                    break
                
                # Handle different filter types
                if isinstance(value, dict):
                    # Range or comparison filters
                    if '$gte' in value and metadata[key] < value['$gte']:
                        matches = False
                        break
                    if '$lte' in value and metadata[key] > value['$lte']:
                        matches = False
                        break
                    if '$eq' in value and metadata[key] != value['$eq']:
                        matches = False
                        break
                    if '$in' in value and metadata[key] not in value['$in']:
                        matches = False
                        break
                elif isinstance(value, str) and isinstance(metadata[key], str):
                    # String contains search
                    if value.lower() not in metadata[key].lower():
                        matches = False
                        break
                else:
                    # Exact match
                    if metadata[key] != value:
                        matches = False
                        break
            
            if matches:
                results.append(metadata)
        
        return results
    
    def get_metadata_by_source(self, source: str) -> List[Dict[str, Any]]:
        """
        Get all metadata entries from a specific source.
        
        Args:
            source: Source identifier
            
        Returns:
            List of metadata entries from the source
        """
        return self.search_metadata({'source': source})
    
    def get_metadata_by_type(self, content_type: str) -> List[Dict[str, Any]]:
        """
        Get all metadata entries of a specific type.
        
        Args:
            content_type: Content type (e.g., 'video', 'document', 'audio')
            
        Returns:
            List of metadata entries of the specified type
        """
        return self.search_metadata({'content_type': content_type})
    
    def get_recent_metadata(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recently added metadata entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of recent metadata entries
        """
        active_metadata = [
            metadata for vector_id, metadata in self.metadata_store.items()
            if vector_id not in self.deleted_ids
        ]
        
        # Sort by added_at timestamp
        sorted_metadata = sorted(
            active_metadata,
            key=lambda x: x.get('added_at', ''),
            reverse=True
        )
        
        return sorted_metadata[:limit]
    
    def get_metadata_count(self) -> int:
        """
        Get count of active metadata entries.
        
        Returns:
            Number of active metadata entries
        """
        return len(self.metadata_store) - len(self.deleted_ids)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get metadata statistics.
        
        Returns:
            Dictionary with statistics
        """
        active_count = self.get_metadata_count()
        
        # Count by content type
        type_counts = {}
        for vector_id, metadata in self.metadata_store.items():
            if vector_id not in self.deleted_ids:
                content_type = metadata.get('content_type', 'unknown')
                type_counts[content_type] = type_counts.get(content_type, 0) + 1
        
        return {
            'total_entries': len(self.metadata_store),
            'active_entries': active_count,
            'deleted_entries': len(self.deleted_ids),
            'total_added': self.total_added,
            'total_deleted': self.total_deleted,
            'type_distribution': type_counts
        }
    
    def save_metadata(self, file_path: Union[str, Path]) -> None:
        """
        Save metadata to file.
        
        Args:
            file_path: Path to save metadata file
        """
        try:
            metadata_data = {
                'metadata_store': self.metadata_store,
                'deleted_ids': list(self.deleted_ids),
                'statistics': {
                    'total_added': self.total_added,
                    'total_deleted': self.total_deleted
                },
                'saved_at': datetime.now().isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Metadata saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
            raise VectorDatabaseError(f"Failed to save metadata: {e}")
    
    def load_metadata(self, file_path: Union[str, Path]) -> None:
        """
        Load metadata from file.
        
        Args:
            file_path: Path to metadata file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                metadata_data = json.load(f)
            
            # Convert string keys back to integers for metadata_store
            self.metadata_store = {
                int(k): v for k, v in metadata_data.get('metadata_store', {}).items()
            }
            
            self.deleted_ids = set(metadata_data.get('deleted_ids', []))
            
            # Load statistics
            stats = metadata_data.get('statistics', {})
            self.total_added = stats.get('total_added', 0)
            self.total_deleted = stats.get('total_deleted', 0)
            
            self.logger.info(f"Metadata loaded from {file_path}")
            self.logger.info(f"Loaded {len(self.metadata_store)} metadata entries")
            
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            raise VectorDatabaseError(f"Failed to load metadata: {e}")
    
    def clear_metadata(self) -> None:
        """Clear all metadata."""
        self.metadata_store.clear()
        self.deleted_ids.clear()
        self.total_added = 0
        self.total_deleted = 0
        self.logger.info("All metadata cleared")
    
    def cleanup_deleted(self) -> int:
        """
        Permanently remove deleted metadata entries.
        
        Returns:
            Number of entries removed
        """
        removed_count = 0
        
        for vector_id in list(self.deleted_ids):
            if vector_id in self.metadata_store:
                del self.metadata_store[vector_id]
                removed_count += 1
        
        self.deleted_ids.clear()
        
        self.logger.info(f"Cleaned up {removed_count} deleted metadata entries")
        return removed_count
    
    def export_metadata(self, file_path: Union[str, Path], include_deleted: bool = False) -> None:
        """
        Export metadata to a human-readable format.
        
        Args:
            file_path: Path to export file
            include_deleted: Whether to include deleted entries
        """
        try:
            export_data = []
            
            for vector_id, metadata in self.metadata_store.items():
                if not include_deleted and vector_id in self.deleted_ids:
                    continue
                
                export_data.append({
                    'vector_id': vector_id,
                    'is_deleted': vector_id in self.deleted_ids,
                    **metadata
                })
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Metadata exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metadata: {e}")
            raise VectorDatabaseError(f"Failed to export metadata: {e}")
