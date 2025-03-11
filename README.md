# 2D Convolution Benchmark: CUDA vs CPU

A high-performance implementation of 2D convolution with GPU (CUDA) and CPU implementations, demonstrating the power of parallel computing for image processing operations.

---

## Project Description

This program implements a **2D convolution operation** - the fundamental building block of Convolutional Neural Networks (CNNs) - with two versions:
- 🚀 **CUDA-accelerated GPU implementation**
- ⏳ **CPU reference implementation**

Key features:
- Zero-padding boundary handling
- Automated performance benchmarking
- Result validation (CPU vs GPU)
- Configurable input/kernel sizes
- Box filter demonstration (replaceable with any kernel)

Designed for `2048x2048` inputs with `5x5` kernels by default, easily modifiable for different use cases.

---

## What is a CNN?

A **Convolutional Neural Network (CNN)** is a deep learning architecture that:
- 🖼️ Specializes in processing grid-like data (images, videos)
- 🔍 Uses convolutional layers to detect spatial patterns
- 🧠 Learns hierarchical features through training
- ⚡ Powers modern computer vision applications:
  - Image recognition
  - Object detection
  - Medical imaging analysis
  - Autonomous vehicle perception

---

## 2D Convolution Explained

The mathematical operation at the heart of CNNs:

```math
Output(x,y) = \sum_{i=-k}^{k} \sum_{j=-k}^{k} Input(x+i,y+j) \cdot Kernel(i,j)


---

## Benchmark Results

### Test Configuration
| Parameter        | Value                               |
|------------------|-------------------------------------|
| Input Size       | 2048x2048 (4.3MP image)             |
| Kernel Size      | 5x5 box filter                      |
| Data Type        | FP32 (Single-precision float)       |
| Runs             | 10 iterations                       |
| Boundary Handling| Zero-padding                        |

### Performance Metrics
| Metric               | GPU               | CPU                | Improvement       |
|----------------------|-------------------|--------------------|-------------------|
| Total Time (10 runs) | 122.249 ms        | 9351.76 ms         | 76.5x             |
| Average per-run      | 12.22 ms          | 935.18 ms          | 76.5x             |
| Memory Throughput    | 672 GB/s          | 45 GB/s            | 14.9x             |
| Pixel Rate           | 350 GPixel/s      | 4.58 GPixel/s      | 76.4x             |
| Max Absolute Error   | 0.0               | 0.0                | -                 |

---

## Hardware Configuration
| Component       | GPU Specs                      | CPU Specs                   |
|-----------------|--------------------------------|-----------------------------|
| Model           | NVIDIA RTX 4070                | Intel i9-13900HX            |
| Memory          | 8GB                            | 16GB                        |
