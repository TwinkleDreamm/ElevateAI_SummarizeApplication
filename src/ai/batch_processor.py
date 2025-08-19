"""
Batch processing utilities for parallel execution tasks.
"""
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Callable, TypeVar, Generic
from dataclasses import dataclass

from src.utils.logger import logger

ResultType = TypeVar('ResultType')

@dataclass
class BatchResult(Generic[ResultType]):
    """Result from processing a batch.
    
    Args:
        batch_index: Index of the batch in the original sequence
        result: The result of processing the batch, type specified by ResultType
    """
    batch_index: int
    result: ResultType

class BatchProcessor(Generic[ResultType]):
    """Parallel batch processor using thread pool.
    
    Type Parameters:
        ResultType: The type of result that will be produced for each batch
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def process_batches(self, 
                       batches: List[List[Dict[str, Any]]], 
                       process_func: Callable[[List[Dict[str, Any]], int], ResultType]) -> List[ResultType]:
        """
        Process batches in parallel while maintaining order.
        
        Args:
            batches: List of batches to process
            process_func: Function that processes a single batch
            
        Returns:
            List of results in original batch order
        """
        try:
            # Submit all batches for processing
            future_to_index = {}
            for i, batch in enumerate(batches):
                future = self.executor.submit(process_func, batch, i)
                future_to_index[future] = i
                
            # Collect results while maintaining order
            results = [None] * len(batches)
            for future in future_to_index:
                try:
                    result = future.result()
                    index = future_to_index[future]
                    results[index] = result
                except Exception as exc:
                    logger.error(f"Error processing batch: {exc}")
                    index = future_to_index[future]
                    results[index] = f"Error processing batch {index}: {str(exc)}"
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            raise
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the executor."""
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error shutting down batch processor: {e}")
