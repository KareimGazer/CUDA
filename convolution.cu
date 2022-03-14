/* 
 * Simple and basic 1D convolution kernel with boundary condition handling.
 * mask_width is assumed to be odd for simplicity and the convolution is symmetric.
 * mask_widh = 2 * n + 1, where n is an integer.
 * The elements to be summed are from N[index-n] to N[index + n]
 * 
 * Author: Kareim Tarek
 * Date: 18th of Feb 2022
*/

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define MAX_MASK_WIDTH 10 // most conv kernels (arrays) will not exceed 10 in each dimension
#define MASK_WIDTH 5
#define WIDTH 100
#define TILE_SIZE 1024
#define OUTPUT_TILE_WIDTH 32
__constant__ float dev_mask_arr[MAX_MASK_WIDTH];


__global__ void convolution_1D_basic_kernel(float* input_arr, float* output_arr, int mask_width, int width);
__global__ void convolution_1D_tiled_kernel(float* input_arr, float* output_arr, int mask_width, int width);
__global__ void convolution_1D_tiled_caching_kernel(float* input_arr, float* output_arr, int mask_width, int width);

int main(){
    float mask_arr[MASK_WIDTH] = {3, 4, 5, 4, 3};
    // to be changed to malloc
    float input_arr[WIDTH] = {1.0};
    float output_arr[WIDTH] = {0.0};


    cudaMemcpyToSymbol(dev_mask_arr, mask_arr, sizeof(float) * MASK_WIDTH);

    return 0;
}

__global__ void convolution_1D_basic_kernel(float* input_arr, float* output_arr, int mask_width, int width){
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  int n = mask_width / 2; // mask_width = 2 * n + 1 
  int starting_index = thread_index - n;
  float acc = 0; // accumulator
  float missing_element_val = 0;
  for(int conv_index = 0; conv_index < mask_width; conv_index++){
    if(conv_index >= 0 && conv_index < width) // use conv weights
      acc += input_arr[starting_index + conv_index] * dev_mask_arr[conv_index];
    else // apply ghost cell values
      acc += input_arr[starting_index + conv_index] * missing_element_val;
  }
  output_arr[thread_index] = acc;
}

/*
 * 1D convolution using tiling and cache memory
 * input: - input_arr: 1D input array, like audio signal
 *        - mask_width: the length of the convolution mask (kernel)
 *        - width: the width of the input and output arrays
 * output: 
 *         - output_arr: 1D output array.
 * NOTE: 
 * 0 =< thread index =< internal tile width
 * TILE_SIZE = blockDim.x
*/
__global__ void convolution_1D_tiled_kernel(float* input_arr, float* output_arr, int mask_width, int width){
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float input_arr_ds[TILE_SIZE + MAX_MASK_WIDTH - 1];
    int n = mask_width / 2; // mask_width = 2 * n + 1
    float missing_element_val = 0;

    // get left halo cells
    int left_halo_index = (blockIdx.x - 1)*blockDim.x + threadIdx.x;
    if(threadIdx.x >= blockDim.x - n ){
        // check if they are ghost cells
        input_arr_ds[threadIdx.x - (blockDim.x - n)] = left_halo_index < 0 ? missing_element_val : input_arr[left_halo_index];
    }

    // get center cells
    input_arr_ds[n + threadIdx.x] = input_arr[thread_index];

    // get right halo cells
    int right_halo_index = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
    if(threadIdx.x < n){
        // check if they are ghost cells
        input_arr_ds[n + blockDim.x + threadIdx.x] = right_halo_index >= width? missing_element_val : input_arr[right_halo_index];
    }

    __syncthreads(); // all data should be loaded before usage

    // do the calculation
    float acc = 0;
    int conv_start_index = threadIdx.x; // not threadIdx.x -n because the array is bigger by 2n
    for(int conv_index = 0; conv_index < mask_width; conv_index++){
        acc += input_arr_ds[conv_start_index + conv_index] * dev_mask_arr[conv_index];
    }
    output_arr[thread_index] = acc;    
}

// where is the if condition to check for the width
__global__ void convolution_1D_tiled_caching_kernel(float* input_arr, float* output_arr, int mask_width, int width){
    int thread_index = blockIdx.x * blockDim.x +threadIdx.x;
    __shared__ float input_arr_ds[TILE_SIZE];
    input_arr_ds[threadIdx.x] = input_arr[thread_index];
    __syncthreads();

    int n = mask_width / 2; // mask_width = 2n + 1
    float missing_element_val = 0;

    int this_tile_start_index = blockIdx.x * blockDim.x;
    int next_tile_start_index = (blockIdx.x + 1) * blockDim.x;

    int start_index = thread_index - n;
    float acc = 0;
    for(int conv_index = 0; conv_index < mask_width; conv_index++){
        int current_index = start_index + conv_index;
        if(current_index >= this_tile_start_index && current_index < next_tile_start_index){
            acc += input_arr_ds[current_index] * dev_mask_arr[conv_index];
        }
        else{ // right or left, halo or ghost
            acc += (current_index < 0 || current_index >= next_tile_start_index) ?
                    missing_element_val * dev_mask_arr[conv_index] : input_arr[current_index] * dev_mask_arr[conv_index];
        }
    }
    output_arr[thread_index] = acc;
}

__global__ void convolution_2D_tiled_kernel(float* input_arr, float* output_arr, int mask_width, 
                                            int height, int pitch, int channels, int width){
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int output_row = blockIdx.y * OUTPUT_TILE_WIDTH + ty;
    int output_col = blockIdx.x * OUTPUT_TILE_WIDTH + tx;
    int input_start_row = output_row - mask_width/2;
    int input_start_col = output_col - mask_width/2;

    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float input_arr_ds[TILE_SIZE + MAX_MASK_WIDTH - 1];
    int n = mask_width / 2; // mask_width = 2 * n + 1
    float missing_element_val = 0;

    // get left halo cells
    int left_halo_index = (blockIdx.x - 1)*blockDim.x + threadIdx.x;
    if(threadIdx.x >= blockDim.x - n ){
        // check if they are ghost cells
        input_arr_ds[threadIdx.x - (blockDim.x - n)] = left_halo_index < 0 ? missing_element_val : input_arr[left_halo_index];
    }

    // get center cells
    input_arr_ds[n + threadIdx.x] = input_arr[thread_index];

    // get right halo cells
    int right_halo_index = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
    if(threadIdx.x < n){
        // check if they are ghost cells
        input_arr_ds[n + blockDim.x + threadIdx.x] = right_halo_index >= width? missing_element_val : input_arr[right_halo_index];
    }

    __syncthreads(); // all data should be loaded before usage

    // do the calculation
    float acc = 0;
    int conv_start_index = threadIdx.x; // not threadIdx.x -n because the array is bigger by 2n
    for(int conv_index = 0; conv_index < mask_width; conv_index++){
        acc += input_arr_ds[conv_start_index + conv_index] * dev_mask_arr[conv_index];
    }
    output_arr[thread_index] = acc;    
}

