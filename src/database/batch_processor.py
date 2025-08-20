"""
Optimized batch processing for vector database operations.
"""
from typing import List, Dict, Any, Optional
import numpy as np
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

from src.utils.logger import logger
from src.utils.performance import measure_performance


class VectorDatabaseBatchProcessor:
    """Optimized batch processor for vector database operations."""
    
    def __init__(self, max_workers: int = 4, batch_size: int = 32):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_queue = Queue()
        self._stop_event = threading.Event()
    
    @measure_performance
    def process_embeddings_batch(self, texts: List[str], embedding_func) -> np.ndarray:
        """
        Process texts into embeddings in parallel batches.
        
        Args:
            texts: List of texts to process
            embedding_func: Function to generate embeddings
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        # Split into batches
        batches = [
            texts[i:i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        
        # Process batches in parallel
        futures = []
        for batch in batches:
            future = self.executor.submit(embedding_func, batch)
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            batch_result = future.result()
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
        
        return np.array(results)
    
    @measure_performance
    def batch_add_to_vectordb(self, 
                            db,
                            texts: List[str],
                            metadata_list: List[Dict[str, Any]],
                            batch_size: Optional[int] = None) -> List[int]:
        """
        Add texts to vector database in optimized batches.
        
        Args:
            db: Vector database instance
            texts: List of texts to add
            metadata_list: List of metadata dictionaries
            batch_size: Optional custom batch size
            
        Returns:
            List of assigned IDs
        """
        if not texts:
            return []
        
        batch_size = batch_size or self.batch_size
        all_ids = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadata = metadata_list[i:i + batch_size]
            
            # Add batch to database
            batch_ids = db.add_to_vectordb(batch_texts, batch_metadata)
            all_ids.extend(batch_ids)
            
            # Log progress
            logger.info(f"Added batch {i//batch_size + 1}/{len(texts)//batch_size + 1}")
        
        return all_ids
    
    def shutdown(self):
        """Shutdown the processor."""
        self._stop_event.set()
        self.executor.shutdown(wait=True)
