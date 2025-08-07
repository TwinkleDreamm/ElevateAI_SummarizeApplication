"""
Performance optimization utilities for ElevateAI.
"""
import time
import psutil
import threading
from typing import Dict, Any, Optional, Callable, List
from functools import wraps
from dataclasses import dataclass
from contextlib import contextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from .logger import logger


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    function_name: str
    timestamp: float


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics: List[PerformanceMetrics] = []
        self.lock = threading.Lock()
    
    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics."""
        with self.lock:
            self.metrics.append(metrics)
            
            # Keep only last 1000 metrics to prevent memory bloat
            if len(self.metrics) > 1000:
                self.metrics = self.metrics[-1000:]
    
    def get_metrics_summary(self, function_name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        with self.lock:
            if function_name:
                filtered_metrics = [m for m in self.metrics if m.function_name == function_name]
            else:
                filtered_metrics = self.metrics
            
            if not filtered_metrics:
                return {}
            
            execution_times = [m.execution_time for m in filtered_metrics]
            memory_usages = [m.memory_usage for m in filtered_metrics]
            cpu_usages = [m.cpu_usage for m in filtered_metrics]
            
            return {
                'count': len(filtered_metrics),
                'avg_execution_time': sum(execution_times) / len(execution_times),
                'max_execution_time': max(execution_times),
                'min_execution_time': min(execution_times),
                'avg_memory_usage': sum(memory_usages) / len(memory_usages),
                'max_memory_usage': max(memory_usages),
                'avg_cpu_usage': sum(cpu_usages) / len(cpu_usages),
                'max_cpu_usage': max(cpu_usages)
            }
    
    def clear_metrics(self) -> None:
        """Clear all recorded metrics."""
        with self.lock:
            self.metrics.clear()


# Global performance monitor
performance_monitor = PerformanceMonitor()


def measure_performance(func: Callable) -> Callable:
    """
    Decorator to measure function performance.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function with performance measurement
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get initial system metrics
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = process.cpu_percent()
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            # Calculate metrics
            execution_time = time.time() - start_time
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            final_cpu = process.cpu_percent()
            
            memory_usage = final_memory - initial_memory
            cpu_usage = final_cpu
            
            # Record metrics
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                function_name=func.__name__,
                timestamp=time.time()
            )
            
            performance_monitor.record_metrics(metrics)
            
            # Log if execution time is significant
            if execution_time > 1.0:
                logger.info(
                    f"Performance: {func.__name__} took {execution_time:.2f}s, "
                    f"memory: {memory_usage:.2f}MB, CPU: {cpu_usage:.1f}%"
                )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
    
    return wrapper


@contextmanager
def performance_context(operation_name: str):
    """
    Context manager for measuring performance of code blocks.
    
    Args:
        operation_name: Name of the operation being measured
    """
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    start_time = time.time()
    
    try:
        yield
        
    finally:
        execution_time = time.time() - start_time
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - initial_memory
        cpu_usage = process.cpu_percent()
        
        metrics = PerformanceMetrics(
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            function_name=operation_name,
            timestamp=time.time()
        )
        
        performance_monitor.record_metrics(metrics)
        
        logger.info(
            f"Performance: {operation_name} took {execution_time:.2f}s, "
            f"memory: {memory_usage:.2f}MB, CPU: {cpu_usage:.1f}%"
        )


class BatchProcessor:
    """Efficient batch processing for large datasets."""
    
    def __init__(self, batch_size: int = 32, max_workers: int = 4):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Size of each batch
            max_workers: Maximum number of worker threads
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max_workers)
    
    def process_batches(self, items: List[Any], process_func: Callable, 
                       use_processes: bool = False) -> List[Any]:
        """
        Process items in batches using threading or multiprocessing.
        
        Args:
            items: Items to process
            process_func: Function to process each batch
            use_processes: Use multiprocessing instead of threading
            
        Returns:
            List of processed results
        """
        if not items:
            return []
        
        # Split items into batches
        batches = [
            items[i:i + self.batch_size] 
            for i in range(0, len(items), self.batch_size)
        ]
        
        executor = self.process_executor if use_processes else self.thread_executor
        
        with performance_context(f"batch_processing_{len(batches)}_batches"):
            try:
                # Submit all batches for processing
                futures = [executor.submit(process_func, batch) for batch in batches]
                
                # Collect results
                results = []
                for future in futures:
                    batch_result = future.result()
                    if isinstance(batch_result, list):
                        results.extend(batch_result)
                    else:
                        results.append(batch_result)
                
                return results
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                raise
    
    def __del__(self):
        """Clean up executors."""
        try:
            self.thread_executor.shutdown(wait=False)
            self.process_executor.shutdown(wait=False)
        except:
            pass


class MemoryOptimizer:
    """Memory optimization utilities."""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    @staticmethod
    def check_memory_threshold(threshold_mb: float = 1000) -> bool:
        """
        Check if memory usage exceeds threshold.
        
        Args:
            threshold_mb: Memory threshold in MB
            
        Returns:
            True if memory usage is below threshold
        """
        memory_usage = MemoryOptimizer.get_memory_usage()
        return memory_usage['rss_mb'] < threshold_mb
    
    @staticmethod
    def optimize_memory():
        """Perform memory optimization."""
        import gc
        
        # Force garbage collection
        collected = gc.collect()
        
        memory_after = MemoryOptimizer.get_memory_usage()
        
        logger.info(
            f"Memory optimization: collected {collected} objects, "
            f"current usage: {memory_after['rss_mb']:.2f}MB"
        )


class APIRateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_minute: Maximum calls per minute
        """
        self.calls_per_minute = calls_per_minute
        self.call_times: List[float] = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        with self.lock:
            now = time.time()
            
            # Remove calls older than 1 minute
            self.call_times = [t for t in self.call_times if now - t < 60]
            
            # Check if we need to wait
            if len(self.call_times) >= self.calls_per_minute:
                # Calculate wait time
                oldest_call = min(self.call_times)
                wait_time = 60 - (now - oldest_call)
                
                if wait_time > 0:
                    logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                    time.sleep(wait_time)
            
            # Record this call
            self.call_times.append(now)
    
    def __call__(self, func: Callable) -> Callable:
        """Use as decorator for rate limiting."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.wait_if_needed()
            return func(*args, **kwargs)
        return wrapper


def optimize_for_production():
    """Apply production optimizations."""
    import gc
    
    # Configure garbage collection for better performance
    gc.set_threshold(700, 10, 10)  # More aggressive GC
    
    # Log optimization status
    logger.info("Production optimizations applied")
    
    return {
        'gc_threshold': gc.get_threshold(),
        'memory_usage': MemoryOptimizer.get_memory_usage()
    }


# Global instances
batch_processor = BatchProcessor()
memory_optimizer = MemoryOptimizer()
api_rate_limiter = APIRateLimiter()
