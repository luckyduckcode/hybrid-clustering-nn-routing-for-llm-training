# Performance Optimization Guide

## Overview

This guide covers optimization strategies for the 1.58-Bit Hybrid LLM Training System. The system supports multiple backends (NumPy, C, C++, GPU) and automatically selects the optimal implementation for your hardware.

## Performance Targets

| Operation | Backend | Speedup | Data Size | Use Case |
|-----------|---------|---------|-----------|----------|
| Quantization | NumPy | 1.0x (baseline) | <100K | Development, small models |
| Quantization | C | 3-10x | 100K-1M | Medium models |
| Quantization | C++ | 5-20x | 1M-100M | Large models |
| Quantization | GPU | 10-50x | >100M | Very large models |
| K-Means | NumPy | 1.0x (baseline) | <10K samples | Development |
| K-Means | C++ | 5-15x | 10K-100K samples | Production |
| K-Means | GPU | 15-40x | >100K samples | Large datasets |

## Quick Start

### 1. Using the Hybrid Dispatcher

```python
from hybrid_dispatcher import create_auto_dispatcher

# Create dispatcher with auto-detection
dispatcher = create_auto_dispatcher()

# Quantization automatically uses best backend
quantized = dispatcher.quantize(values)

# K-Means automatically uses best backend
labels, centroids = dispatcher.kmeans(data, n_clusters=10)

# Get performance metrics
metrics = dispatcher.get_metrics()
print(f"Total operations: {metrics['total_operations']}")
print(f"Backend summary: {metrics['backend_summary']}")
```

### 2. Benchmarking Your System

```python
from benchmarks import PerformanceBenchmark

benchmark = PerformanceBenchmark()
report = benchmark.run_all_benchmarks()
print(report)
```

## Backend Selection Guide

### When to Use Each Backend

**NumPy (Always Available)**
- ✓ Development and prototyping
- ✓ Data size < 100K elements
- ✓ Single training iteration
- ✓ No compilation/setup needed
- ✗ Production with large datasets
- ✗ Real-time inference

**C Backend (3-10x faster)**
- ✓ Medium data (100K-1M elements)
- ✓ Production deployment
- ✓ Cache-efficient SIMD operations
- ✓ Low memory footprint
- ✗ Very large datasets
- ✗ GPU-accelerated systems

**C++ Backend (5-20x faster)**
- ✓ Large data (1M-100M elements)
- ✓ Complex algorithm kernels
- ✓ Multi-threaded workloads
- ✓ Production inference
- ✗ GPU-available systems
- ✗ Real-time latency-critical applications

**GPU Backend (10-50x faster)**
- ✓ Very large data (>100M elements)
- ✓ Batch processing
- ✓ Parallel computing
- ✓ Real-time training
- ✗ Systems without CUDA
- ✗ Embedded deployment

## Installation & Setup

### 1. Core Installation (Required)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install NumPy (required)
pip install numpy

# Optional: matplotlib for visualization
pip install matplotlib
```

### 2. C Backend Setup

**Prerequisites:**
- GCC or Clang compiler
- Make utility

**Windows (MinGW):**
```bash
# Install MinGW
# Download from https://www.mingw-w64.org/
# Add to PATH

# Compile
gcc -O3 -march=native -c quantization.c -o quantization.o
gcc -O3 -march=native -shared quantization.o -o libhybrid_c.so
```

**Linux/macOS:**
```bash
# Compile with SIMD support
gcc -O3 -march=native -msse4.2 -mavx -mavx2 -c quantization.c
gcc -O3 -march=native -msse4.2 -mavx -mavx2 -shared quantization.o -o libhybrid_c.so

# Or with clang
clang -O3 -march=native -c quantization.c
clang -O3 -march=native -shared quantization.o -o libhybrid_c.so
```

### 3. C++ Backend Setup

**Prerequisites:**
- GCC, Clang, or MSVC
- Make utility

**Linux/macOS:**
```bash
# Compile C++ backend
g++ -std=c++17 -O3 -march=native -fPIC -c quantization.cpp
g++ -std=c++17 -O3 -march=native -fPIC -shared quantization.o -o libhybrid.so

# Or with OpenMP for parallelization
g++ -std=c++17 -O3 -march=native -fPIC -fopenmp -c quantization.cpp
g++ -std=c++17 -O3 -march=native -fPIC -fopenmp -shared quantization.o -o libhybrid.so
```

**Windows (MSVC):**
```batch
# Compile with Visual Studio
cl /O2 /arch:AVX2 /c quantization.cpp
link /DLL quantization.obj /out:libhybrid.dll

# Or with MinGW
g++ -std=c++17 -O3 -march=native -shared quantization.cpp -o libhybrid.dll
```

### 4. GPU Backend Setup (CUDA/CuPy)

**Prerequisites:**
- NVIDIA GPU
- CUDA Toolkit 11.0+
- cuDNN (optional, for ML operations)

**Installation:**
```bash
# Install CuPy for CUDA 11.8
pip install cupy-cuda11x

# Or build from source for custom CUDA
pip install cupy --no-binary cupy

# Verify installation
python -c "import cupy; print(cupy.cuda.Device())"
```

**Docker Setup (Recommended):**
```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3-pip
RUN pip install numpy cupy-cuda11x
COPY . /app
WORKDIR /app
```

## Performance Tuning

### 1. Quantization Optimization

```python
from hybrid_dispatcher import HybridDispatcher, BackendConfig

# Configure for GPU priority
config = BackendConfig()
config.prefer_gpu = True
config.min_size_for_gpu = 500_000  # Lower threshold
config.verbose = True

dispatcher = HybridDispatcher(config)

# Quantize with benchmarking
quantized = dispatcher.quantize(data, benchmark=True)

# Get metrics
metrics = dispatcher.get_metrics()
```

### 2. K-Means Optimization

```python
from clustering import KMeansClustering, DataClustering

# Use reduced precision for faster convergence
clustering = DataClustering(
    n_clusters=10,
    use_int8=True,  # 8-bit clustering
    n_init=1,       # Single initialization
    max_iter=50     # Fewer iterations
)

# Get cluster assignments
labels = clustering.cluster_embeddings(data)
```

### 3. Training Pipeline Optimization

```python
from training_system import HybridLLMTrainer, TrainingConfig

# Optimize for speed over accuracy
config = TrainingConfig(
    batch_size=256,              # Larger batches
    learning_rate=0.01,
    max_epochs=10,
    n_clusters=5,                # Fewer clusters
    quantize_weights=True,
    quantize_gradients=True,
    use_adaptive_lr=True,
    checkpoint_dir='checkpoints'
)

trainer = HybridLLMTrainer(config)
# Train with auto backend selection
trainer.train(train_data)
```

## Memory Optimization

### 1. Reduce Quantization Precision

```python
from quantization import AdaptiveQuantizer

# Start with lower precision
quantizer = AdaptiveQuantizer(
    target_bits=1.58,
    enable_adaptive=True  # Adapt based on data distribution
)

# Adaptive precision reduces memory further
quantized = quantizer.quantize(values)
```

### 2. Use Clustering for Data Reduction

```python
from clustering import DataClustering

# Cluster data to reduce effective size
clustering = DataClustering(n_clusters=20)
cluster_ids = clustering.cluster_embeddings(large_data)

# Train on cluster representatives (20x smaller)
centroids = clustering.get_centroids()
model.train(centroids)
```

### 3. Batch Processing Strategy

```python
import numpy as np
from hybrid_dispatcher import HybridDispatcher

dispatcher = HybridDispatcher()

# Process large data in batches
batch_size = 100_000
results = []

for batch in batches:  # batches of 100K elements
    result = dispatcher.quantize(batch)
    results.append(result)

final_result = np.concatenate(results)
```

## Bottleneck Analysis

### 1. Identify Slow Operations

```python
from hybrid_dispatcher import AdaptiveDispatcher

# Enable metrics tracking
dispatcher = AdaptiveDispatcher()
dispatcher.config.benchmark_mode = True

# Run training
train_model()

# Get bottleneck analysis
metrics = dispatcher.get_metrics()
for backend, stats in metrics['backend_summary'].items():
    print(f"{backend}:")
    print(f"  Count: {stats['count']}")
    print(f"  Avg time: {stats['avg_time_ms']:.2f}ms")
    print(f"  Total time: {stats['total_time_ms']:.2f}ms")

# Get recommendations
recommendation = dispatcher.recommend_compilation()
print(f"Recommendation: {recommendation}")
```

### 2. Profile with Python Profiler

```python
import cProfile
import pstats

# Profile training
profiler = cProfile.Profile()
profiler.enable()

train_model()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### 3. Memory Profiling

```python
from memory_profiler import profile

@profile
def train_batch(data):
    quantizer = dispatcher.select_quantizer(data.size)[0]
    quantized = quantizer.quantize(data)
    return quantized

# Run with memory profiler
# python -m memory_profiler script.py
```

## Deployment Recommendations

### Development Mode
```python
from hybrid_dispatcher import HybridDispatcher, BackendConfig

config = BackendConfig()
config.prefer_gpu = False      # Skip GPU for debugging
config.prefer_cpp = False      # Skip C++ for simplicity
config.verbose = True

dispatcher = HybridDispatcher(config)
```

### Production Mode (CPUs)
```python
config = BackendConfig()
config.prefer_gpu = False
config.prefer_cpp = True       # Use optimized C++
config.min_size_for_cpp = 10_000  # Lower threshold
config.verbose = False         # Minimal logging

dispatcher = HybridDispatcher(config)
```

### Production Mode (GPU)
```python
config = BackendConfig()
config.prefer_gpu = True       # Prioritize GPU
config.min_size_for_gpu = 500_000  # More aggressive
config.verbose = False

dispatcher = HybridDispatcher(config)
```

### Embedded/Edge Deployment
```python
config = BackendConfig()
config.prefer_gpu = False
config.prefer_cpp = False
config.prefer_c = True         # Use C for small footprint

dispatcher = HybridDispatcher(config)
```

## Troubleshooting

### GPU Not Detected

```python
from gpu_backend import HAS_GPU, get_gpu_info

if not HAS_GPU:
    print("GPU not available")
    print("Install CuPy: pip install cupy-cuda11x")
else:
    info = get_gpu_info()
    print(f"GPU Device: {info['device']}")
    print(f"Total Memory: {info['total_memory_gb']:.2f} GB")
```

### C/C++ Library Not Found

```bash
# Check library existence
# Linux/macOS
find . -name "libhybrid*.so"

# Windows
where libhybrid*.dll

# Set library path
export LD_LIBRARY_PATH=/path/to/libs:$LD_LIBRARY_PATH

# Or rebuild
./build.sh  # See build instructions above
```

### OutOfMemory Errors

1. **Reduce batch size:**
   ```python
   config.batch_size = 64  # From 256
   ```

2. **Use quantization:**
   ```python
   config.quantize_weights = True
   config.quantize_gradients = True
   ```

3. **Use clustering:**
   ```python
   config.n_clusters = 20  # Increase clustering
   ```

## Benchmarking Results (Example)

```
Quantization (1M float32 values):
  NumPy: 45.23ms | Memory: 4.2MB | Throughput: 22,106 elem/ms
  C++: 3.12ms | Memory: 3.8MB | Speedup: 14.5x
  GPU: 1.02ms | Memory: 512.3MB | Speedup: 44.3x

K-Means (10K samples x 100 features):
  NumPy: 234.5ms | Memory: 8.5MB
  C++: 28.4ms | Memory: 8.2MB | Speedup: 8.3x
  GPU: 12.1ms | Memory: 128.5MB | Speedup: 19.4x

Matrix Operations (100K x 100 dot product):
  NumPy: 156.2ms | Throughput: 64,000 elem/ms
  C: 24.1ms | Speedup: 6.5x
  GPU: 8.3ms | Speedup: 18.8x
```

## References

- [NumPy Performance Guide](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [SIMD Optimization Guide](https://en.wikipedia.org/wiki/SIMD)
- [CuPy Documentation](https://docs.cupy.dev/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

## Support

For issues or questions:
1. Check this guide's troubleshooting section
2. Run benchmark suite to identify bottlenecks
3. Enable verbose logging: `config.verbose = True`
4. Review optimization recommendations from adaptive dispatcher
