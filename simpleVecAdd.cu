
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

/* cudaDeviceReset must be called before exiting in order for profilingand
   tracing tools such as Nsight and Visual Profiler to show complete traces. */
#define CUDA_DEVICE_RESET                           \
    cudaStatus = cudaDeviceReset();                 \
    if (cudaStatus != cudaSuccess)                  \
    {                                               \
        fprintf(stderr, "cudaDeviceReset failed!"); \
        return 1;                                   \
    }

#define CUDA_CHECK_ERROR(cudaStatus)                                                         \
    if (cudaStatus != cudaSuccess)                                                           \
    {                                                                                        \
        printf("%s in %s in line %d\n", cudaGetErrorString(cudaStatus), __FILE__, __LINE__); \
        return 1;                                                                            \
    }

// cudaDeviceSynchronize waits for the kernel to finish, and returns
// any errors encountered during the launch.
#define CUDA_DEVICE_SYNCH                                                                                         \
    cudaStatus = cudaDeviceSynchronize();                                                                         \
    if (cudaStatus != cudaSuccess)                                                                                \
    {                                                                                                             \
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); \
        return 1;                                                                                                 \
    }

__global__ void addVec(int *C, int *A, int *B, unsigned int size);

int main()
{
    const int arraySize = 1000;
    const float block_dim = 32.0;
    cudaError_t cudaStatus; // need for all cuda macors
    size_t vec_size = arraySize * sizeof(int);
    // C = A + B
    int h_A[arraySize] = {1};
    int h_B[arraySize] = {2};
    int h_C[arraySize] = {0};

    // init
    for (int i = 0; i < arraySize; i++)
    {
        h_A[i] = i * 10;
        h_B[i] = 1;
    }

    int *d_A = 0;
    int *d_B = 0;
    int *d_C = 0;
    cudaStatus = cudaMalloc((void **)&d_A, vec_size);
    CUDA_CHECK_ERROR(cudaStatus)

    cudaStatus = cudaMalloc((void **)&d_B, vec_size);
    CUDA_CHECK_ERROR(cudaStatus)

    cudaStatus = cudaMalloc((void **)&d_C, vec_size);
    CUDA_CHECK_ERROR(cudaStatus)

    cudaStatus = cudaMemcpy(d_A, h_A, vec_size, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR(cudaStatus)
    cudaStatus = cudaMemcpy(d_B, h_B, vec_size, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR(cudaStatus)

    addVec<<<ceil(arraySize / block_dim), block_dim>>>(d_C, d_A, d_B, vec_size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    CUDA_CHECK_ERROR(cudaStatus)

    CUDA_DEVICE_SYNCH

    cudaStatus = cudaMemcpy(h_C, d_C, vec_size, cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR(cudaStatus)

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    CUDA_DEVICE_RESET

    for (int i = 0; i < arraySize; i++)
    {
        printf("%d ", h_C[i]);
    }
    printf("\n");
    return 0;
}

__global__ void addVec(int *C, int *A, int *B, unsigned int size)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        C[index] = A[index] + B[index];
}
