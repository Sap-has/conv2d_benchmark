#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T result, const char* const func, const char* const file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << result
            << " \"" << func << "\"" << std::endl;
        exit(1);
    }
}

// CUDA 2D Convolution Kernel
__global__ void conv2DKernel(float* input, float* kernel, float* output,
    int inputWidth, int inputHeight,
    int kernelWidth, int kernelHeight) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= inputWidth || row >= inputHeight) return;

    int halfKernelW = kernelWidth / 2;
    int halfKernelH = kernelHeight / 2;
    float sum = 0.0f;

    for (int ky = -halfKernelH; ky <= halfKernelH; ++ky) {
        for (int kx = -halfKernelW; kx <= halfKernelW; ++kx) {
            int inputRow = row + ky;
            int inputCol = col + kx;

            if (inputRow >= 0 && inputRow < inputHeight &&
                inputCol >= 0 && inputCol < inputWidth) {
                sum += input[inputRow * inputWidth + inputCol] *
                    kernel[(ky + halfKernelH) * kernelWidth + (kx + halfKernelW)];
            }
        }
    }
    output[row * inputWidth + col] = sum;
}

// CPU Convolution Implementation
void cpuConv2D(const std::vector<float>& input, const std::vector<float>& kernel,
    std::vector<float>& output, int inputWidth, int inputHeight,
    int kernelWidth, int kernelHeight) {
    int halfKernelW = kernelWidth / 2;
    int halfKernelH = kernelHeight / 2;

    for (int row = 0; row < inputHeight; ++row) {
        for (int col = 0; col < inputWidth; ++col) {
            float sum = 0.0f;

            for (int ky = -halfKernelH; ky <= halfKernelH; ++ky) {
                for (int kx = -halfKernelW; kx <= halfKernelW; ++kx) {
                    int inputRow = row + ky;
                    int inputCol = col + kx;

                    if (inputRow >= 0 && inputRow < inputHeight &&
                        inputCol >= 0 && inputCol < inputWidth) {
                        sum += input[inputRow * inputWidth + inputCol] *
                            kernel[(ky + halfKernelH) * kernelWidth + (kx + halfKernelW)];
                    }
                }
            }
            output[row * inputWidth + col] = sum;
        }
    }
}

int main() {
    // Configuration
    const int inputWidth = 2048;
    const int inputHeight = 2048;
    const int kernelSize = 5;
    const int numRuns = 10;

    // Create input and kernel
    std::vector<float> input(inputWidth * inputHeight);
    std::vector<float> kernel(kernelSize * kernelSize, 1.0f);  // Box filter
    std::vector<float> cpuOutput(inputWidth * inputHeight);
    std::vector<float> gpuOutput(inputWidth * inputHeight);

    // Initialize input with random values
    for (auto& val : input) val = static_cast<float>(rand()) / RAND_MAX;

    // Allocate GPU memory
    float* d_input, * d_kernel, * d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_kernel, kernel.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, input.size() * sizeof(float)));

    // Copy data to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, kernel.data(), kernel.size() * sizeof(float),
        cudaMemcpyHostToDevice));

    // Configure CUDA kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((inputWidth + blockSize.x - 1) / blockSize.x,
        (inputHeight + blockSize.y - 1) / blockSize.y);

    // Warm-up and time GPU
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < numRuns; ++i) {
        conv2DKernel <<<gridSize, blockSize >>> (d_input, d_kernel, d_output,
            inputWidth, inputHeight,
            kernelSize, kernelSize);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float gpuTime;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpuTime, start, stop));

    // Copy result back
    CHECK_CUDA_ERROR(cudaMemcpy(gpuOutput.data(), d_output, input.size() * sizeof(float),
        cudaMemcpyDeviceToHost));

    // Time CPU
    auto cpuStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numRuns; ++i) {
        cpuConv2D(input, kernel, cpuOutput, inputWidth, inputHeight, kernelSize, kernelSize);
    }
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    float cpuTime = std::chrono::duration<float, std::milli>(cpuEnd - cpuStart).count();

    // Validate results
    float maxError = 0.0f;
    for (size_t i = 0; i < cpuOutput.size(); ++i) {
        maxError = fmax(maxError, fabs(cpuOutput[i] - gpuOutput[i]));
    }
    std::cout << "Maximum absolute error: " << maxError << std::endl;

    // Print results
    std::cout << "GPU Time (" << numRuns << " runs): " << gpuTime << " ms\n";
    std::cout << "CPU Time (" << numRuns << " runs): " << cpuTime << " ms\n";
    std::cout << "Speedup: " << cpuTime / gpuTime << "x\n";

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}