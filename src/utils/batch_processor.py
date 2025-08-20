"""
Batch processing utilities with thread pool optimization.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Any, Callable, TypeVar, Optional
from dataclasses import dataclass
from pathlib import Path
import threading
import queue
import time

from .logger import logger
from .performance import measure_performance

T = TypeVar('T')
R = TypeVar('R')

@dataclass
class ProcessingResult:
    """Result of a processing operation."""
    success: bool
    result: Any = None
    error: Optional[Exception] = None

class BatchProcessor:
    """Efficient batch processing with thread pool."""
    
    def __init__(self, max_workers: int = 4, batch_size: int = 10):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.results_queue = queue.Queue()
        self._stop_event = threading.Event()
    
    @measure_performance
    def process_items(self, items: List[T], process_func: Callable[[T], R]) -> List[ProcessingResult]:
        """
        Process a list of items in parallel using thread pool.
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            
        Returns:
            List of ProcessingResult objects
        """
        if not items:
            return []
        
        results: List[ProcessingResult] = []
        futures = {}
        
        # Submit items in batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            for item in batch:
                future = self.executor.submit(self._safe_process, process_func, item)
                futures[future] = item
            
            # Wait for batch completion
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                # Log progress
                if len(results) % 10 == 0:
                    logger.info(f"Processed {len(results)}/{len(items)} items")
        
        return results
    
    def _safe_process(self, func: Callable[[T], R], item: T) -> ProcessingResult:
        """Safely execute processing function with error handling."""
        try:
            result = func(item)
            return ProcessingResult(success=True, result=result)
        except Exception as e:
            logger.error(f"Processing failed for item {item}: {e}")
            return ProcessingResult(success=False, error=e)
    
    def shutdown(self):
        """Shutdown the processor."""
        self._stop_event.set()
        self.executor.shutdown(wait=True)

class AsyncBatchProcessor:
    """Asynchronous batch processor with callbacks."""
    
    def __init__(self, max_workers: int = 4):
        self.processor = BatchProcessor(max_workers=max_workers)
        self.processing_thread: Optional[threading.Thread] = None
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._callback = None
    
    def start(self, callback: Callable[[ProcessingResult], None]):
        """Start async processing with callback."""
        self._callback = callback
        self.processing_thread = threading.Thread(target=self._process_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def add_item(self, item: Any, process_func: Callable):
        """Add item for async processing."""
        self._queue.put((item, process_func))
    
    def _process_queue(self):
        while not self._stop_event.is_set():
            try:
                item, process_func = self._queue.get(timeout=1)
                result = self.processor._safe_process(process_func, item)
                if self._callback:
                    self._callback(result)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Async processing error: {e}")
    
    def stop(self):
        """Stop async processing."""
        self._stop_event.set()
        if self.processing_thread:
            self.processing_thread.join()
        self.processor.shutdown()
