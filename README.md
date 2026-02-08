# Parallel Iterative SpMV Computer

A high-performance parallel implementation of Sparse Matrix-Vector Multiplication (SpMV) using OpenMP, featuring multiple optimization strategies and research-driven performance analysis.

## Project Overview

This project implements and analyzes parallel SpMV (Sparse Matrix-Vector Multiplication) algorithms using OpenMP. The implementation includes multiple optimization approaches including row-based parallelization, BCSR (Block Compressed Sparse Row) with dense blocks, and BCSR with sparse blocks. Through systematic experimentation and analysis, significant performance improvements were achieved over baseline serial implementations.

## Key Research Contributions

### Algorithmic Optimizations
1. **Row-based Parallelization**: Implements OpenMP parallel for loops with dynamic scheduling for load balancing across matrix rows
2. **BCSR Dense Block Optimization**: Uses Block Compressed Sparse Row format with dense blocks to improve cache locality
3. **BCSR Sparse Block Optimization**: Implements BCSR with sparse blocks for memory efficiency while maintaining performance
4. **Memory Access Optimization**: Optimizes data layout and access patterns for better cache utilization

### Performance Innovations
- **Cache-aware blocking** for improved spatial locality
- **Dynamic workload distribution** for irregular sparse matrices
- **Vectorization-friendly code structure**
- **Reduced synchronization overhead** through careful parallel design

## Project Structure

```
.
├── README.md                           # Project documentation
├── parco2025-hw2.pdf                   # Assignment specification
├── Report_Parallel_Computing_HW2.pdf   # Detailed research report
├── parco2025-hw2-code/                 # Source code and test data
│   ├── Makefile                        # Build configuration
│   ├── check.py                        # Result verification script
│   ├── serial.cpp                      # Baseline serial implementation
│   ├── parallel_row.cpp                # Row-based parallel implementation
│   ├── parallel_BCSR_dense.cpp         # BCSR dense block implementation
│   ├── parallel_BCSR_sparse.cpp        # BCSR sparse block implementation
│   ├── run.serial                      # Serial execution script
│   ├── run.parallel_row                # Row parallel execution script
│   ├── run.BCSR_dense                  # BCSR dense execution script
│   ├── run.BCSR_sparse                 # BCSR sparse execution script
│   └── input*.csr                      # Test matrix files (1-5)
└── submission/                         # Submission directory
    ├── parallel_row.cpp                # Row-based parallel implementation
    ├── parallel_BCSR_sparse.cpp        # BCSR sparse block implementation
    ├── serial.cpp                      # Serial implementation
    ├── run.serial                      # Serial execution script
    ├── run.parallel_row                # Row parallel execution script
    └── run.BCSR_sparse                 # BCSR sparse execution script
```

## Efficiency Features

### 1. Memory Efficiency
- **CSR format optimization** reduces memory overhead for sparse matrices
- **Blocking techniques** improve cache utilization for both dense and sparse blocks
- **Data structure alignment** for better vectorization support

### 2. Parallel Efficiency
- **Dynamic scheduling** with empirically tuned chunk sizes for irregular workloads
- **Minimal synchronization** through independent row/block processing
- **Load balancing** across threads for matrices with varying row densities
- **Thread affinity optimization** for better cache utilization

### 3. Computational Optimizations
- **Loop unrolling** for inner product computations
- **Prefetching hints** for better memory access patterns
- **Compiler optimization** flags tuned for target architecture
- **Branch reduction** in critical computation loops

## Build and Run

### Build
```bash
cd parco2025-hw2-code
make
```
Compiles all components with optimization flags: `-O3 -fopenmp -std=c++11`

### Execution Examples
```bash
# Serial baseline
./serial input1.csr

# Row-based parallel (8 threads)
export OMP_NUM_THREADS=8
./parallel_row input1.csr

# BCSR dense block parallel (8 threads)
export OMP_NUM_THREADS=8
./parallel_BCSR_dense input1.csr

# BCSR sparse block parallel (8 threads)
export OMP_NUM_THREADS=8
./parallel_BCSR_sparse input1.csr
```

### Automated Testing
Use the provided run scripts for consistent testing:
```bash
# Run serial version
./run.serial input1.csr

# Run parallel versions
./run.parallel_row input1.csr
./run.BCSR_dense input1.csr
./run.BCSR_sparse input1.csr
```

## Benchmark Results

The implementation was tested on multiple sparse matrices with varying characteristics:
- **Small matrices** (input1.csr): For algorithm validation
- **Medium matrices** (input2-4.csr): For performance scaling analysis
- **Large matrices** (input5.csr): For memory and scalability testing

Performance metrics evaluated:
- **Strong scaling**: Fixed problem size with increasing threads
- **Weak scaling**: Problem size grows with thread count
- **Memory bandwidth utilization**: For different block sizes
- **Cache effects**: Impact of cache sizes on different optimization strategies

## Research Methodology

### 1. Baseline Analysis
- Profiled serial SpMV implementation to identify bottlenecks
- Analyzed memory access patterns with performance profiling tools
- Identified irregular memory access as main performance limitation

### 2. Parallelization Strategies
- Implemented multiple parallel approaches (row-based, block-based)
- Measured overhead of different synchronization methods
- Optimized block sizes based on empirical testing and cache characteristics

### 3. Memory Optimization
- Compared different sparse matrix formats (CSR, BCSR)
- Implemented blocking techniques for better cache utilization
- Optimized data layout for vectorized operations

### 4. Performance Tuning
- Systematically tested compiler optimization flags
- Evaluated thread affinity and scheduling parameters
- Tuned block sizes for different matrix characteristics

## Performance Metrics

The implementation achieves:
- **Significant speedup** over serial implementation across all test matrices
- **Good scaling behavior** up to available core counts
- **Reduced memory bandwidth pressure** through cache optimization
- **Consistent correctness** across all test cases

## Technical Details

### Algorithm Complexity
- **Time**: O(nnz) with parallel speedup, where nnz is number of non-zeros
- **Space**: O(n + nnz) for CSR format, plus blocking overhead
- **Parallel overhead**: Dependent on load balancing and synchronization

### Hardware Considerations
- Optimized for modern multi-core CPUs with hierarchical caches
- Cache-aware design for L1/L2/L3 cache hierarchies
- Vectorization-friendly code structure
- NUMA-aware for multi-socket systems

## Author

**Leslie Wang** - Research and implementation of optimized parallel SpMV algorithms.

Repository: https://github.com/cwangPKU/Parallel-Iterative-SpMV-Computer

## License

Research code - see included report for detailed analysis and findings.