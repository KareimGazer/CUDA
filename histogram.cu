#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <unistd.h>

#define HEIGHT 10
#define WIDTH 100
#define LEVELS 10

__global__ void histogramPrivatized(unsigned char* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins));


int main(){

    uint8_t input[WIDTH], result[LEVELS];

    uint8_t * dev_input = 0;
    uint8_t * dev_result = 0;

	for(int i=0; i<WIDTH; i++){
		input[i] = i / 10;
	}

    cudaMalloc((void**)&dev_input, WIDTH * sizeof(uint8_t));
    cudaMalloc((void**)&dev_result, WIDTH * sizeof(uint8_t));

    cudaMemcpy(dev_input, input, WIDTH * sizeof(uint8_t), cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(WIDTH/32.0), 1, 1);
    dim3 dimBlock(32, 32, 1);

    histogramPrivatized << <dimGrid2, dimBlock2 >> > (dev_output, dev_input, dev_rndArray);

    cudaDeviceSynchronize();
    cudaMemcpy(result, dev_result, WIDTH * sizeof(uint8_t), cudaMemcpyDeviceToHost);

	for(int i=0; i<LEVELS; i++){
		printf("%d , ", result[i]);
	}
	printf("\n");

    cudaFree(dev_input);
    cudaFree(dev_result);
    return 0;
}

__global__ void histogramPrivatized(unsigned char* input, unsigned int* result, unsigned int num_elements, int input_len){
	int threadIndex =  blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ uint8_t local_histogram[]; // shared memory

	// loading the values into shared memory
	for(int inputIndex = threadIdx.x; inputIndex < input_len; inputIndex +=blockDim.x) {
		local_histogram[inputIndex] = 0;
	}
	__syncthreads();

	// local histogram operation
	for(unsigned int i = tid; i < num_elements; i += blockDim.x*gridDim.x) {
		int alphabet_position = buffer[i] – “a”;
		if (alphabet_position >= 0 && alpha_position < 26) atomicAdd(&(histo_s[alphabet_position/4]), 1);
	}
	__syncthreads();

	// writing to global memory
	for(unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x) {
		atomicAdd(&(histo[binIdx]), histo_s[binIdx]);
	}
}