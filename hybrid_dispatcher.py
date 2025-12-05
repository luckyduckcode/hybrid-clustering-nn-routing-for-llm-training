"""
Hybrid Backend Dispatcher for 1.58-Bit LLM Training

Automatically selects the optimal backend (NumPy, C, C++, or GPU)
based on data size, computation type, and available hardware.

Selection Strategy:
- Small data (<1M elements): NumPy (overhead not worth it)
- Medium data (1M-100M): C or C++ (if available)
- Large data (>100M): GPU (if available), else C++
- Critical paths: Always use compiled code if available
"""

import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any
import warnings
import time

# Import all backends
from gpu_backend import (
    GPUQuantizer, GPUKMeans, GPUMatrixOps, GPUTrainingBackend,
    HAS_GPU
)
from cpp_backend import (
    CPPQuantizer, CPPKMeans, CPPOptimizationStep, CPPHybridOptimizer,
    _HAS_CPP
)
from c_backend import (
    CQuantizer, CMatrixOps, CConstraintOperator, CHybridOptimizer,
    _HAS_C
)


class BackendConfig:
    """Configuration for backend selection."""
    
    def __init__(self):
        """Initialize backend configuration."""
        self.prefer_gpu = True
        self.prefer_cpp = True
        self.prefer_c = True
        self.min_size_for_gpu = 1_000_000      # 1M elements
        self.min_size_for_cpp = 100_000        # 100K elements
        self.min_size_for_c = 10_000           # 10K elements
        self.verbose = False
        self.benchmark_mode = False


class BackendMetrics:
    """Track backend performance metrics."""
    
    def __init__(self):
        """Initialize metrics."""
        self.backend_times = {}      # backend -> list of times
        self.backend_selections = {} # backend -> count
        self.total_operations = 0
    
    def record_operation(self, backend: str, time_ms: float):
        """Record operation timing."""
        if backend not in self.backend_times:
            self.backend_times[backend] = []
        self.backend_times[backend].append(time_ms)
        
        if backend not in self.backend_selections:
            self.backend_selections[backend] = 0
        self.backend_selections[backend] += 1
        
        self.total_operations += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {}
        
        for backend, times in self.backend_times.items():
            summary[backend] = {
                'count': len(times),
                'avg_time_ms': float(np.mean(times)),
                'min_time_ms': float(np.min(times)),
                'max_time_ms': float(np.max(times)),
                'total_time_ms': float(np.sum(times)),
            }
        
        return summary


class HybridDispatcher:
    """Intelligent backend dispatcher."""
    
    def __init__(self, config: Optional[BackendConfig] = None):
        """
        Initialize hybrid dispatcher.
        
        Args:
            config: Backend configuration
        """
        self.config = config or BackendConfig()
        self.metrics = BackendMetrics()
        
        # Initialize all backends
        self.numpy_backend = True  # Always available
        self.gpu_backend = GPUTrainingBackend() if HAS_GPU else None
        self.cpp_backend = CPPHybridOptimizer() if _HAS_CPP else None
        self.c_backend = CHybridOptimizer() if _HAS_C else None
        
        self._log_initialization()
    
    def _log_initialization(self):
        """Log available backends."""
        if self.config.verbose:
            print("Backend Availability:")
            print(f"  NumPy: Always available")
            print(f"  GPU: {HAS_GPU}")
            print(f"  C++: {_HAS_CPP}")
            print(f"  C: {_HAS_C}")
    
    def select_quantizer(self, data_size: int, operation: str = "quantize") -> Tuple[Any, str]:
        """
        Select best quantizer backend.
        
        Args:
            data_size: Number of elements
            operation: Type of operation ('quantize', 'gradient', etc.)
            
        Returns:
            Tuple of (quantizer, backend_name)
        """
        # GPU for large data
        if (self.config.prefer_gpu and HAS_GPU and 
            data_size >= self.config.min_size_for_gpu):
            return GPUQuantizer(), "GPU"
        
        # C++ for medium data
        if (self.config.prefer_cpp and _HAS_CPP and 
            data_size >= self.config.min_size_for_cpp):
            return CPPQuantizer(), "C++"
        
        # C for smaller data
        if (self.config.prefer_c and _HAS_C and 
            data_size >= self.config.min_size_for_c):
            return CQuantizer(), "C"
        
        # NumPy fallback
        return CPPQuantizer(use_cpp=False), "NumPy"
    
    def select_kmeans(self, data_size: int, n_features: int) -> Tuple[Any, str]:
        """
        Select best K-Means backend.
        
        Args:
            data_size: Number of samples
            n_features: Number of features
            
        Returns:
            Tuple of (kmeans, backend_name)
        """
        total_elements = data_size * n_features
        
        # GPU for very large data
        if (self.config.prefer_gpu and HAS_GPU and 
            total_elements >= self.config.min_size_for_gpu * 10):
            return GPUKMeans(n_clusters=10), "GPU"
        
        # C++ for large data
        if (self.config.prefer_cpp and _HAS_CPP and 
            total_elements >= self.config.min_size_for_cpp):
            return CPPKMeans(n_clusters=10), "C++"
        
        # NumPy for smaller data
        from clustering import KMeansClustering
        return KMeansClustering(n_clusters=10), "NumPy"
    
    def select_optimizer(self) -> Tuple[Any, str]:
        """
        Select best optimizer backend.
        
        Returns:
            Tuple of (optimizer, backend_name)
        """
        # Prefer GPU if available
        if self.config.prefer_gpu and self.gpu_backend:
            return self.gpu_backend, "GPU"
        
        # Prefer C++ if available
        if self.config.prefer_cpp and self.cpp_backend:
            return self.cpp_backend, "C++"
        
        # Prefer C if available
        if self.config.prefer_c and self.c_backend:
            return self.c_backend, "C"
        
        # NumPy fallback
        from training_system import AdaptiveConstrainedOptimizer
        return AdaptiveConstrainedOptimizer(), "NumPy"
    
    def quantize(self, values: np.ndarray, benchmark: bool = False) -> np.ndarray:
        """
        Dispatch quantization to best backend.
        
        Args:
            values: Values to quantize
            benchmark: Enable benchmarking
            
        Returns:
            Quantized values
        """
        quantizer, backend = self.select_quantizer(values.size)
        
        if self.config.benchmark_mode or benchmark:
            start = time.time()
        
        result = quantizer.quantize(values)
        
        if self.config.benchmark_mode or benchmark:
            elapsed_ms = (time.time() - start) * 1000
            self.metrics.record_operation(backend, elapsed_ms)
            if self.config.verbose:
                print(f"  Quantize ({values.size} elems): {backend} ({elapsed_ms:.2f}ms)")
        
        return result
    
    def kmeans(self, data: np.ndarray, n_clusters: int, 
               benchmark: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dispatch K-Means to best backend.
        
        Args:
            data: Input data
            n_clusters: Number of clusters
            benchmark: Enable benchmarking
            
        Returns:
            Tuple of (labels, centroids)
        """
        kmeans, backend = self.select_kmeans(len(data), data.shape[1])
        
        if self.config.benchmark_mode or benchmark:
            start = time.time()
        
        kmeans.fit(data)
        labels = kmeans.labels
        centroids = kmeans.centroids
        
        if self.config.benchmark_mode or benchmark:
            elapsed_ms = (time.time() - start) * 1000
            self.metrics.record_operation(backend, elapsed_ms)
            if self.config.verbose:
                print(f"  K-Means ({len(data)} samples): {backend} ({elapsed_ms:.2f}ms)")
        
        return labels, centroids
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about available backends."""
        return {
            'numpy': True,
            'gpu': HAS_GPU,
            'cpp': _HAS_CPP,
            'c': _HAS_C,
            'preferred_order': self._get_preference_order(),
        }
    
    def _get_preference_order(self) -> list:
        """Get backend preference order."""
        order = []
        
        if self.config.prefer_gpu and HAS_GPU:
            order.append('GPU')
        if self.config.prefer_cpp and _HAS_CPP:
            order.append('C++')
        if self.config.prefer_c and _HAS_C:
            order.append('C')
        
        order.append('NumPy')
        return order
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            'total_operations': self.metrics.total_operations,
            'backend_summary': self.metrics.get_summary(),
        }


class AdaptiveDispatcher(HybridDispatcher):
    """Advanced dispatcher with learning and adaptation."""
    
    def __init__(self, config: Optional[BackendConfig] = None):
        """
        Initialize adaptive dispatcher.
        
        Args:
            config: Backend configuration
        """
        super().__init__(config)
        self.backend_performance = {}  # backend -> [times]
        self.size_thresholds = {}      # operation_type -> size thresholds
    
    def update_thresholds(self):
        """Learn optimal size thresholds from metrics."""
        summary = self.metrics.get_summary()
        
        # If C++ is faster than GPU for some sizes, adjust threshold
        if 'C++' in summary and 'GPU' in summary:
            cpp_avg = summary['C++']['avg_time_ms']
            gpu_avg = summary['GPU']['avg_time_ms']
            
            if cpp_avg < gpu_avg:
                # C++ is competitive with GPU, don't use GPU as much
                self.config.min_size_for_gpu *= 1.5
    
    def recommend_compilation(self) -> str:
        """Recommend which backends to compile."""
        summary = self.metrics.get_summary()
        
        if not summary:
            return "Insufficient data for recommendations"
        
        recommendations = []
        
        # If NumPy is slow, compile C/C++
        if 'NumPy' in summary:
            numpy_avg = summary['NumPy']['avg_time_ms']
            
            if 'C' not in summary and 'C++' not in summary:
                recommendations.append("Compile C/C++ backend for 3-20x speedup")
        
        # If GPU isn't available but there's large data
        if not HAS_GPU and self.metrics.total_operations > 100:
            recommendations.append("GPU not available - install CUDA for 10-50x speedup on large data")
        
        return " | ".join(recommendations) if recommendations else "Current setup is optimal"


def create_auto_dispatcher() -> HybridDispatcher:
    """Create dispatcher with optimal settings."""
    config = BackendConfig()
    config.verbose = True
    config.benchmark_mode = False
    
    return HybridDispatcher(config)


def benchmark_all_backends():
    """Benchmark all available backends."""
    print("\n" + "="*70)
    print("Backend Performance Comparison")
    print("="*70)
    
    dispatcher = HybridDispatcher()
    
    # Quantization benchmark
    print("\nQuantization (1M float32 values):")
    data = np.random.randn(1000000).astype(np.float32)
    result = dispatcher.quantize(data, benchmark=True)
    
    # K-Means benchmark
    print("\nK-Means (10K samples x 100 features):")
    data = np.random.randn(10000, 100)
    labels, centroids = dispatcher.kmeans(data, n_clusters=10, benchmark=True)
    
    # Print metrics
    print("\nPerformance Metrics:")
    metrics = dispatcher.get_metrics()
    for backend, stats in metrics['backend_summary'].items():
        print(f"\n  {backend}:")
        print(f"    Operations: {stats['count']}")
        print(f"    Avg time: {stats['avg_time_ms']:.2f}ms")
        print(f"    Total time: {stats['total_time_ms']:.2f}ms")


if __name__ == "__main__":
    print("Hybrid Backend Dispatcher for 1.58-Bit LLM Training")
    print("=" * 70)
    
    dispatcher = create_auto_dispatcher()
    
    info = dispatcher.get_backend_info()
    print("\nAvailable Backends:")
    for backend, available in info.items():
        if backend != 'preferred_order':
            print(f"  {backend}: {available}")
    
    print("\nPreferred Order:")
    for i, backend in enumerate(info['preferred_order'], 1):
        print(f"  {i}. {backend}")
    
    # Run benchmarks
    benchmark_all_backends()
