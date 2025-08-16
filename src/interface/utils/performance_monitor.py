"""
Performance monitoring utility for ElevateAI Notebooks.
Helps identify bottlenecks and optimize loading times.
"""
import time
import functools
from typing import Callable, Any
import streamlit as st


class PerformanceMonitor:
    """Monitor performance of various operations."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(duration)
            del self.start_times[operation]
            return duration
        return 0.0
    
    def get_average_time(self, operation: str) -> float:
        """Get average time for an operation."""
        if operation in self.metrics and self.metrics[operation]:
            return sum(self.metrics[operation]) / len(self.metrics[operation])
        return 0.0
    
    def display_metrics(self):
        """Display performance metrics in Streamlit."""
        if not self.metrics:
            st.info("No performance metrics collected yet.")
            return
        
        st.subheader("📊 Performance Metrics")
        
        for operation, times in self.metrics.items():
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            count = len(times)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Operation", operation)
            with col2:
                st.metric("Avg Time", f"{avg_time:.3f}s")
            with col3:
                st.metric("Min Time", f"{min_time:.3f}s")
            with col4:
                st.metric("Max Time", f"{max_time:.3f}s")
            
            st.caption(f"Executed {count} times")
            st.divider()


def monitor_performance(operation_name: str):
    """Decorator to monitor performance of functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get or create performance monitor in session state
            if 'performance_monitor' not in st.session_state:
                st.session_state.performance_monitor = PerformanceMonitor()
            
            monitor = st.session_state.performance_monitor
            monitor.start_timer(operation_name)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = monitor.end_timer(operation_name)
                # Log performance for debugging
                if duration > 1.0:  # Log slow operations
                    st.warning(f"⚠️ Slow operation: {operation_name} took {duration:.2f}s")
        
        return wrapper
    return decorator


def cache_expensive_operation(cache_key: str, ttl_seconds: int = 300):
    """Decorator to cache expensive operations with TTL."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create cache key from function name and arguments
            full_cache_key = f"{cache_key}_{hash(str(args) + str(kwargs))}"
            
            # Check if result is cached and not expired
            if full_cache_key in st.session_state:
                cache_entry = st.session_state[full_cache_key]
                if time.time() - cache_entry['timestamp'] < ttl_seconds:
                    return cache_entry['result']
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            st.session_state[full_cache_key] = {
                'result': result,
                'timestamp': time.time()
            }
            
            return result
        
        return wrapper
    return decorator


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def show_performance_widget():
    """Show performance monitoring widget in sidebar."""
    with st.sidebar:
        st.subheader("⚡ Performance Monitor")
        
        if st.button("📊 Show Metrics"):
            performance_monitor.display_metrics()
        
        if st.button("🗑️ Clear Metrics"):
            performance_monitor.metrics.clear()
            st.success("Performance metrics cleared!")
        
        # Show current cache status
        cache_keys = [k for k in st.session_state.keys() if k.startswith('notebooks_cache_')]
        if cache_keys:
            st.info(f"📦 {len(cache_keys)} notebook caches active")
        
        # Show memory usage hints
        st.caption("💡 Performance Tips:")
        st.caption("• Use pagination for large lists")
        st.caption("• Enable lazy loading")
        st.caption("• Cache expensive operations")
        st.caption("• Monitor slow operations")
