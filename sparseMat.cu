/*
implementation of sparse matrix-vector multiplication and accumulation (SpMV)
based on the Compressed Sparse Row (CSR) format
Ax+y = 0
A: sparse cofficient matrix
x: unknowns vector
y: constants vector
used in solving inverse matrix of A using
*/

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// #define WIDTH 100
// #define TILE_SIZE 1024
// #define OUTPUT_TILE_WIDTH 32

/*
test: Ix+y ==> x = -y
*/

/*
sequential CPU code for SpMV
input: parameters
    data: data array of A elements with non zero values
    rowPtr: array of the starting indicies of rows
    rowsNum: the number of rows in the sparse matrix
    colIdx: array of the starting indicies of cols
    x: array of x elements
output:
    y: array of y elements after matrix-vector multiplication
*/
void seq_SpMV_CSR(float * data, int * rowPtr, int rowsNum, int* colIdx, int * x, int * y){
    // rows summation is indep
    for (int row = 0; row < rowsNum; row++) {
        float dot = 0;
        int rowStart = rowPtr[row];
        int rowEnd = rowPtr[row+1];

        // can be divided into coalesced data, aggregated and then atomicadd
        for (int elemIdx = rowStart; elemIdx < rowEnd; elemIdx++) {
            dot += data[elemIdx] * x[colIdx[elemIdx]];
        }
        y[row] += dot;
    }
}


/*
simple parallel GPU code for SpMV
two shortcomings: - The kernel does not make coalesced memory accesses
                  - potential to incur significant control flow divergence in all warps
input: parameters
    data: data array of A elements with non zero values
    rowPtr: array of the starting indicies of rows
    rowsNum: the number of rows in the sparse matrix
    colIdx: array of the starting indicies of cols
    x: array of x elements
output:
    y: array of y elements after matrix-vector multiplication
*/
__global__ void par_SpMV_CSR(int numRows, float *data, int *colIdx,
 int *rowPtr, float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        float dot = 0;
        int rowStart = rowPtr[row];
        int rowEnd = rowPtr[row+1];
        for (int elemIdx = rowStart; elemIdx < rowEnd; elemIdx++) {
            dot += data[elemIdx] * x[colIdx[elemIdx]];
        }
        y[row] += dot;
    }
}

int main(){
    return 0;
}





