/*
* comparison of matrix multiplication between CPU and GPU both using global memroy and shared memroy .
*/
#include <stdio.h>
#include <stdint.h>
#include <chrono>
#include <iostream>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include <device_functions.h>
//#include <cuda_runtime_api.h>

#define BLOCK_SIZE 32
#define TILE_WIDTH BLOCK_SIZE

// Matrices dimension
#define I 128
#define J 128
#define K 128

/*
* Multiplies matrices M and N into P
* M i X j input
* N j X k input
* P i X k output
*/

/*
* @discription matrix multiplication kernel using global memroy
* @input M, N input matrices
* @output P output matrix
*/
__global__ void multiplyKernel(float * M, float * N, float * P);

/*
* @discription matrix multiplication kernel using shared memroy
* @input M, N input matrices
* @output P output matrix
*/
__global__ void multiplyTiledKernel(float* M, float* N, float* P);

int main() {

	float M[I][J], N[J][K], Pcpu[I][K], Pgpu[I][K], Ptiled[I][K];
	float *d_M, *d_N, *d_P, *d_Ptiled;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float outerTimes[4];

	// initiallization
	for (int i = 0; i < I; i++) {
		for (int k = 0; k < K; k++) {
			Pcpu[i][k] = 0;
			Pgpu[i][k] = 0;
			Ptiled[i][k] = 0;

			for (int j = 0; j < J; j++) {
				M[i][j] = -j * k + i;
				N[j][k] = j * i - k;
			}
		}
	}

	cudaMalloc((void**)&d_M, sizeof(float) * I * J);
	cudaMalloc((void**)&d_N, sizeof(float) * J * K);
	cudaMalloc((void**)&d_P, sizeof(float) * I * K);
	cudaMalloc((void**)&d_Ptiled, sizeof(float) * I * K);

	cudaEventRecord(start, 0);

	cudaMemcpy(d_M, M, sizeof(float) * I * J, cudaMemcpyHostToDevice);
	cudaMemcpy(d_N, N, sizeof(float) * J * K, cudaMemcpyHostToDevice);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&outerTimes[0], start, stop);

	dim3 dimGrid(ceil(K / (float) BLOCK_SIZE), ceil(I / (float) BLOCK_SIZE), 1);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

	cudaEventRecord(start, 0);

	multiplyKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P);

	cudaEventRecord(stop, 0);

	cudaMemcpy(Pgpu, d_P, sizeof(float) * I * K, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&outerTimes[1], start, stop);

	cudaEventRecord(start, 0);

	multiplyTiledKernel << <dimGrid, dimBlock >> > (d_M, d_N, d_Ptiled);
	cudaMemcpy(Ptiled, d_Ptiled, sizeof(float) * I * K, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&outerTimes[2], start, stop);

	cudaEventRecord(start, 0);
	auto start_time = std::chrono::high_resolution_clock::now();
	// multiplying with CPU
	for (int i = 0; i < I; ++i)
		for (int j = 0; j < J; ++j)
			for (int k = 0; k < K; ++k){
				Pcpu[i][k] += M[i][j] * N[j][k];
			}
	auto end_time = std::chrono::high_resolution_clock::now();
	auto time = end_time - start_time;

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&outerTimes[3], start, stop);
	
	//printing results
	printf("these time intervals were take by CUDA event API\n");
	printf("Time taken  moving M and N from host to device = %f ms\n", outerTimes[0]);
	printf("Time taken in multiplying using global memory  = %f ms\n", outerTimes[1]);
	printf("Time taken in multiplying using shared memory (Tiling)  = %f ms\n", outerTimes[2]);
	printf("Time taken by CPU (computed cuda events)= %f ms\n", outerTimes[3]);
	std::cout << "\nTime taken by CPU using C++ chronous lib is  " << time / std::chrono::milliseconds(1) << " ms" << std::endl;

	//printing arrays
	
	printf("\nMatrix M: \n");
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			printf("%f ", M[i][j]);
		}
		printf("\n ");
	}

	printf("\nMatrix N: \n");
	for (int j = 0; j < J; j++) {
		for (int k = 0; k < K; k++) {
			printf("%f ", N[j][k]);
		}
		printf("\n ");
	}

	printf("\n Comparing Results: \n");

	printf("\nMatrix Pcpu: \n");
	for (int i = 0; i < I; i++) {
		for (int k = 0; k < K; k++) {
			printf("%f ", Pcpu[i][k]);
		}
		printf("\n ");
	}

	printf("\nMatrix Pgpu: \n");
	for (int i = 0; i < I; i++) {
		for (int k = 0; k < K; k++) {
			printf("%f ", Pgpu[i][k]);
		}
		printf("\n ");
	}

	printf("\nMatrix Ptiled: \n");
	for (int i = 0; i < I; i++) {
		for (int k = 0; k < K; k++) {
			printf("%f ", Ptiled[i][k]);
		}
		printf("\n ");
	}
	

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_M);
	cudaFree(d_N);
	cudaFree(d_P);
	cudaFree(d_Ptiled);

	return 0;
}

__global__ void multiplyKernel(float* M, float* N, float* P) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < I && col < K) {
		float sum = 0.0;
		for (int elementIdx = 0; elementIdx < J; elementIdx++) {
			sum += M[row * J + elementIdx] * N[elementIdx * K + col];
		}
		P[row * K + col] = sum;
	}
}

__global__ void multiplyTiledKernel(float* M, float* N, float* P) {
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x, tx = threadIdx.x, by = blockIdx.y, ty = threadIdx.y;
	int row = by * BLOCK_SIZE + ty; // used macro for faster execution
	int col = bx * BLOCK_SIZE + tx;

	float sum = 0.0;
	int width = I > K ? I: K;
	int phases = ceil(width / (float)TILE_WIDTH);
	for (int phase = 0; phase < phases; phase++) {
		if (row < I && (phase * TILE_WIDTH + tx < J)) Mds[ty][tx] = M[row * J + phase * TILE_WIDTH + tx];
		else Mds[ty][tx] = 0.0;
		if (col < K && ((phase * TILE_WIDTH + ty) < J)) Nds[ty][tx] = N[(phase * TILE_WIDTH + ty) * K + col];
		else Nds[ty][tx] = 0.0;
		__syncthreads();
		for (int elementIdx = 0; elementIdx < TILE_WIDTH; elementIdx++) {
			sum += Mds[ty][elementIdx] * Nds[elementIdx][tx];
		}
		__syncthreads();
	}
	if(row < I && col < K) P[row * K + col] = sum;
}