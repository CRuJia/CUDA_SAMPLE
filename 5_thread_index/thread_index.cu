#include <cuda_runtime.h>
#include <stdio.h>

#include "utils.h"

__global__ void PrintThreadIndex(float* A, const int nx, const int ny) {
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int idx = iy * nx + ix;
  printf(
      "thread_id(%d,%d) block_id(%d,%d) coordinate(%d,%d)"
      "global index %2d ival %f\n",
      threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);
}

int main() {
  initDevice(0);
  int nx = 8;
  int ny = 6;
  int nxy = nx * ny;
  int nBytes = nxy * sizeof(float);

  // Malloc
  float* a_h = (float*)malloc(nBytes);
  initialData(a_h, nxy);
  printMaxtrix(a_h, nx, ny);

  // cudaMalloc
  float* a_d = NULL;
  CHECK(cudaMalloc((void**)&a_d, nBytes));

  CHECK(cudaMemcpy(a_d, a_h, nBytes, cudaMemcpyHostToDevice));

  dim3 block(4, 2);
  dim3 grid((nx - 1) / block.x + 1, (ny - 1) / block.y + 1);

  PrintThreadIndex<<<grid, block>>>(a_d, nx, ny);

  CHECK(cudaDeviceSynchronize());
  CHECK(cudaFree(a_d));
  free(a_h);

  cudaDeviceReset();
  return 0;
}