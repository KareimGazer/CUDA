# CUDA
implementing several image processing functions using CUDA

## Histogram Computation
used PRIVATIZATION via shared memory along with atomic operations.
Histogram computation causes many race conditions which need atomic operations but these interensic functions leads to high memory latency because it serializes the work to solve race conditions which results in decreased throughput.
by directing the atomic operations to shared memeory and then copying the results back to memory latency is reduced significantly.

## Intensity Transformations
- Histogram Equalization
- Image Thresholding
- Bit-plane Slicing: compute / memeory access ratio is 8 for 8-bit gray scale images using global memory

## Sparse Matrix Computation

## Graph Search

## Merge Sort

## Convolution
used cached for

## Matrix Multiplication
shared memory with any dimension
