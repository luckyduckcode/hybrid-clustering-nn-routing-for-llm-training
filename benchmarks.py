"""
Comprehensive Performance Benchmarking Suite

Measures performance of NumPy vs C vs C++ vs GPU backends
across quantization, clustering, and optimization operations.

Reports:
- Execution time per operation
- Memory usage patterns
- Speedup ratios
- Optimal data size ranges for each backend
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import time
import tracemalloc
from dataclasses import dataclass
import json


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    operation: str
    backend: str
    data_size: int
    time_ms: float
    memory_mb: float
    throughput_elements_per_ms: float


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self):
        """Initialize benchmarking suite."""
        self.results: List[BenchmarkResult] = []
        self.config = {
            'quantization_sizes': [10_000, 100_000, 1_000_000, 10_000_000],
            'kmeans_configs': [
                (100, 32),    # 100 samples x 32 features
                (1000, 64),   # 1K samples x 64 features
                (10000, 128), # 10K samples x 128 features
            ],
            'matrix_sizes': [
                (1000, 1000),
                (10000, 1000),
                (100000, 100),
            ],
            'iterations': 3,  # Average over N runs
        }
    
    def benchmark_quantization(self) -> List[BenchmarkResult]:
        """Benchmark quantization across all backends."""
        print("\n" + "="*70)
        print("QUANTIZATION BENCHMARKS")
        print("="*70)
        
        results = []
        
        for size in self.config['quantization_sizes']:
            print(f"\nData size: {size:,} elements")
            
            # Generate test data
            data = np.random.randn(size).astype(np.float32)
            
            # NumPy baseline
            numpy_result = self._benchmark_numpy_quantize(data)
            results.append(numpy_result)
            print(f"  NumPy: {numpy_result.time_ms:.2f}ms | "
                  f"Memory: {numpy_result.memory_mb:.2f}MB | "
                  f"Throughput: {numpy_result.throughput_elements_per_ms:.0f} elem/ms")
            
            # GPU if available
            gpu_result = self._benchmark_gpu_quantize(data)
            if gpu_result:
                results.append(gpu_result)
                speedup = numpy_result.time_ms / gpu_result.time_ms
                print(f"  GPU: {gpu_result.time_ms:.2f}ms | "
                      f"Memory: {gpu_result.memory_mb:.2f}MB | "
                      f"Speedup: {speedup:.2f}x")
            
            # C++ if available
            cpp_result = self._benchmark_cpp_quantize(data)
            if cpp_result:
                results.append(cpp_result)
                speedup = numpy_result.time_ms / cpp_result.time_ms
                print(f"  C++: {cpp_result.time_ms:.2f}ms | "
                      f"Memory: {cpp_result.memory_mb:.2f}MB | "
                      f"Speedup: {speedup:.2f}x")
            
            # C if available
            c_result = self._benchmark_c_quantize(data)
            if c_result:
                results.append(c_result)
                speedup = numpy_result.time_ms / c_result.time_ms
                print(f"  C: {c_result.time_ms:.2f}ms | "
                      f"Memory: {c_result.memory_mb:.2f}MB | "
                      f"Speedup: {speedup:.2f}x")
        
        return results
    
    def benchmark_kmeans(self) -> List[BenchmarkResult]:
        """Benchmark K-Means across all backends."""
        print("\n" + "="*70)
        print("K-MEANS CLUSTERING BENCHMARKS")
        print("="*70)
        
        results = []
        
        for n_samples, n_features in self.config['kmeans_configs']:
            print(f"\nData: {n_samples:,} samples x {n_features} features "
                  f"({n_samples * n_features:,} elements)")
            
            # Generate test data
            data = np.random.randn(n_samples, n_features).astype(np.float32)
            
            # NumPy baseline
            numpy_result = self._benchmark_numpy_kmeans(data, n_clusters=10)
            results.append(numpy_result)
            print(f"  NumPy: {numpy_result.time_ms:.2f}ms | "
                  f"Memory: {numpy_result.memory_mb:.2f}MB")
            
            # GPU if available
            gpu_result = self._benchmark_gpu_kmeans(data, n_clusters=10)
            if gpu_result:
                results.append(gpu_result)
                speedup = numpy_result.time_ms / gpu_result.time_ms
                print(f"  GPU: {gpu_result.time_ms:.2f}ms | "
                      f"Memory: {gpu_result.memory_mb:.2f}MB | "
                      f"Speedup: {speedup:.2f}x")
            
            # C++ if available
            cpp_result = self._benchmark_cpp_kmeans(data, n_clusters=10)
            if cpp_result:
                results.append(cpp_result)
                speedup = numpy_result.time_ms / cpp_result.time_ms
                print(f"  C++: {cpp_result.time_ms:.2f}ms | "
                      f"Memory: {cpp_result.memory_mb:.2f}MB | "
                      f"Speedup: {speedup:.2f}x")
        
        return results
    
    def benchmark_matrix_operations(self) -> List[BenchmarkResult]:
        """Benchmark matrix operations across backends."""
        print("\n" + "="*70)
        print("MATRIX OPERATION BENCHMARKS (Dot Product)")
        print("="*70)
        
        results = []
        
        for rows, cols in self.config['matrix_sizes']:
            print(f"\nMatrix: {rows:,} x {cols} ({rows * cols:,} elements)")
            
            # Generate test data
            A = np.random.randn(rows, cols).astype(np.float32)
            B = np.random.randn(cols, cols).astype(np.float32)
            
            # NumPy baseline
            numpy_result = self._benchmark_numpy_matmul(A, B)
            results.append(numpy_result)
            print(f"  NumPy: {numpy_result.time_ms:.2f}ms | "
                  f"Throughput: {numpy_result.throughput_elements_per_ms:.0f} elem/ms")
            
            # GPU if available
            gpu_result = self._benchmark_gpu_matmul(A, B)
            if gpu_result:
                results.append(gpu_result)
                speedup = numpy_result.time_ms / gpu_result.time_ms
                print(f"  GPU: {gpu_result.time_ms:.2f}ms | "
                      f"Speedup: {speedup:.2f}x")
            
            # C if available
            c_result = self._benchmark_c_matmul(A, B)
            if c_result:
                results.append(c_result)
                speedup = numpy_result.time_ms / c_result.time_ms
                print(f"  C: {c_result.time_ms:.2f}ms | "
                      f"Speedup: {speedup:.2f}x")
        
        return results
    
    def _benchmark_numpy_quantize(self, data: np.ndarray) -> BenchmarkResult:
        """Benchmark NumPy quantization."""
        from quantization import Quantizer158Bit
        
        quantizer = Quantizer158Bit()
        
        tracemalloc.start()
        
        times = []
        for _ in range(self.config['iterations']):
            start = time.time()
            _ = quantizer.quantize(data)
            times.append((time.time() - start) * 1000)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        avg_time = np.mean(times)
        memory_mb = peak / 1024 / 1024
        throughput = data.size / avg_time
        
        return BenchmarkResult(
            operation='quantize',
            backend='NumPy',
            data_size=data.size,
            time_ms=avg_time,
            memory_mb=memory_mb,
            throughput_elements_per_ms=throughput,
        )
    
    def _benchmark_gpu_quantize(self, data: np.ndarray) -> BenchmarkResult:
        """Benchmark GPU quantization."""
        try:
            from gpu_backend import GPUQuantizer
            quantizer = GPUQuantizer()
            
            tracemalloc.start()
            
            times = []
            for _ in range(self.config['iterations']):
                start = time.time()
                _ = quantizer.quantize(data)
                times.append((time.time() - start) * 1000)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            avg_time = np.mean(times)
            memory_mb = peak / 1024 / 1024
            throughput = data.size / avg_time
            
            return BenchmarkResult(
                operation='quantize',
                backend='GPU',
                data_size=data.size,
                time_ms=avg_time,
                memory_mb=memory_mb,
                throughput_elements_per_ms=throughput,
            )
        except:
            return None
    
    def _benchmark_cpp_quantize(self, data: np.ndarray) -> BenchmarkResult:
        """Benchmark C++ quantization."""
        try:
            from cpp_backend import CPPQuantizer
            quantizer = CPPQuantizer()
            
            tracemalloc.start()
            
            times = []
            for _ in range(self.config['iterations']):
                start = time.time()
                _ = quantizer.quantize(data)
                times.append((time.time() - start) * 1000)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            avg_time = np.mean(times)
            memory_mb = peak / 1024 / 1024
            throughput = data.size / avg_time
            
            return BenchmarkResult(
                operation='quantize',
                backend='C++',
                data_size=data.size,
                time_ms=avg_time,
                memory_mb=memory_mb,
                throughput_elements_per_ms=throughput,
            )
        except:
            return None
    
    def _benchmark_c_quantize(self, data: np.ndarray) -> BenchmarkResult:
        """Benchmark C quantization."""
        try:
            from c_backend import CQuantizer
            quantizer = CQuantizer()
            
            tracemalloc.start()
            
            times = []
            for _ in range(self.config['iterations']):
                start = time.time()
                _ = quantizer.quantize(data)
                times.append((time.time() - start) * 1000)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            avg_time = np.mean(times)
            memory_mb = peak / 1024 / 1024
            throughput = data.size / avg_time
            
            return BenchmarkResult(
                operation='quantize',
                backend='C',
                data_size=data.size,
                time_ms=avg_time,
                memory_mb=memory_mb,
                throughput_elements_per_ms=throughput,
            )
        except:
            return None
    
    def _benchmark_numpy_kmeans(self, data: np.ndarray, n_clusters: int) -> BenchmarkResult:
        """Benchmark NumPy K-Means."""
        from clustering import KMeansClustering
        
        kmeans = KMeansClustering(n_clusters=n_clusters)
        
        tracemalloc.start()
        
        times = []
        for _ in range(self.config['iterations']):
            start = time.time()
            kmeans.fit(data)
            times.append((time.time() - start) * 1000)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        avg_time = np.mean(times)
        memory_mb = peak / 1024 / 1024
        
        return BenchmarkResult(
            operation='kmeans',
            backend='NumPy',
            data_size=data.size,
            time_ms=avg_time,
            memory_mb=memory_mb,
            throughput_elements_per_ms=data.size / avg_time,
        )
    
    def _benchmark_gpu_kmeans(self, data: np.ndarray, n_clusters: int) -> BenchmarkResult:
        """Benchmark GPU K-Means."""
        try:
            from gpu_backend import GPUKMeans
            kmeans = GPUKMeans(n_clusters=n_clusters)
            
            tracemalloc.start()
            
            times = []
            for _ in range(self.config['iterations']):
                start = time.time()
                kmeans.fit(data)
                times.append((time.time() - start) * 1000)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            avg_time = np.mean(times)
            memory_mb = peak / 1024 / 1024
            
            return BenchmarkResult(
                operation='kmeans',
                backend='GPU',
                data_size=data.size,
                time_ms=avg_time,
                memory_mb=memory_mb,
                throughput_elements_per_ms=data.size / avg_time,
            )
        except:
            return None
    
    def _benchmark_cpp_kmeans(self, data: np.ndarray, n_clusters: int) -> BenchmarkResult:
        """Benchmark C++ K-Means."""
        try:
            from cpp_backend import CPPKMeans
            kmeans = CPPKMeans(n_clusters=n_clusters)
            
            tracemalloc.start()
            
            times = []
            for _ in range(self.config['iterations']):
                start = time.time()
                kmeans.fit(data)
                times.append((time.time() - start) * 1000)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            avg_time = np.mean(times)
            memory_mb = peak / 1024 / 1024
            
            return BenchmarkResult(
                operation='kmeans',
                backend='C++',
                data_size=data.size,
                time_ms=avg_time,
                memory_mb=memory_mb,
                throughput_elements_per_ms=data.size / avg_time,
            )
        except:
            return None
    
    def _benchmark_numpy_matmul(self, A: np.ndarray, B: np.ndarray) -> BenchmarkResult:
        """Benchmark NumPy matrix multiplication."""
        tracemalloc.start()
        
        times = []
        for _ in range(self.config['iterations']):
            start = time.time()
            _ = np.dot(A, B)
            times.append((time.time() - start) * 1000)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        avg_time = np.mean(times)
        memory_mb = peak / 1024 / 1024
        throughput = (A.shape[0] * A.shape[1] * B.shape[1]) / avg_time
        
        return BenchmarkResult(
            operation='matmul',
            backend='NumPy',
            data_size=A.size + B.size,
            time_ms=avg_time,
            memory_mb=memory_mb,
            throughput_elements_per_ms=throughput,
        )
    
    def _benchmark_gpu_matmul(self, A: np.ndarray, B: np.ndarray) -> BenchmarkResult:
        """Benchmark GPU matrix multiplication."""
        try:
            from gpu_backend import GPUMatrixOps
            ops = GPUMatrixOps()
            
            tracemalloc.start()
            
            times = []
            for _ in range(self.config['iterations']):
                start = time.time()
                _ = ops.dot(A, B)
                times.append((time.time() - start) * 1000)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            avg_time = np.mean(times)
            memory_mb = peak / 1024 / 1024
            throughput = (A.shape[0] * A.shape[1] * B.shape[1]) / avg_time
            
            return BenchmarkResult(
                operation='matmul',
                backend='GPU',
                data_size=A.size + B.size,
                time_ms=avg_time,
                memory_mb=memory_mb,
                throughput_elements_per_ms=throughput,
            )
        except:
            return None
    
    def _benchmark_c_matmul(self, A: np.ndarray, B: np.ndarray) -> BenchmarkResult:
        """Benchmark C matrix multiplication."""
        try:
            from c_backend import CMatrixOps
            ops = CMatrixOps()
            
            tracemalloc.start()
            
            times = []
            for _ in range(self.config['iterations']):
                start = time.time()
                _ = ops.dot(A, B)
                times.append((time.time() - start) * 1000)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            avg_time = np.mean(times)
            memory_mb = peak / 1024 / 1024
            throughput = (A.shape[0] * A.shape[1] * B.shape[1]) / avg_time
            
            return BenchmarkResult(
                operation='matmul',
                backend='C',
                data_size=A.size + B.size,
                time_ms=avg_time,
                memory_mb=memory_mb,
                throughput_elements_per_ms=throughput,
            )
        except:
            return None
    
    def generate_report(self) -> str:
        """Generate comprehensive benchmark report."""
        report = []
        report.append("\n" + "="*70)
        report.append("PERFORMANCE BENCHMARK REPORT")
        report.append("="*70)
        
        # Summary statistics
        report.append("\nSummary Statistics:")
        report.append(f"  Total benchmarks: {len(self.results)}")
        report.append(f"  Operations tested: {len(set(r.operation for r in self.results))}")
        report.append(f"  Backends tested: {len(set(r.backend for r in self.results))}")
        
        # Speedup analysis
        report.append("\nSpeedup Analysis:")
        operations = set(r.operation for r in self.results)
        
        for op in operations:
            op_results = [r for r in self.results if r.operation == op]
            numpy_results = [r for r in op_results if r.backend == 'NumPy']
            
            if numpy_results:
                report.append(f"\n  {op.upper()}:")
                numpy_time = np.mean([r.time_ms for r in numpy_results])
                
                for backend in ['GPU', 'C++', 'C']:
                    backend_results = [r for r in op_results if r.backend == backend]
                    if backend_results:
                        backend_time = np.mean([r.time_ms for r in backend_results])
                        speedup = numpy_time / backend_time
                        report.append(f"    {backend}: {speedup:.2f}x faster than NumPy")
        
        return "\n".join(report)
    
    def run_all_benchmarks(self) -> str:
        """Run all benchmarks and return report."""
        self.results.extend(self.benchmark_quantization())
        self.results.extend(self.benchmark_kmeans())
        self.results.extend(self.benchmark_matrix_operations())
        
        return self.generate_report()


if __name__ == "__main__":
    print("1.58-Bit LLM Training System - Performance Benchmarks")
    print("=" * 70)
    
    benchmark = PerformanceBenchmark()
    report = benchmark.run_all_benchmarks()
    print(report)
    
    print("\n" + "="*70)
    print("Benchmarking complete!")
