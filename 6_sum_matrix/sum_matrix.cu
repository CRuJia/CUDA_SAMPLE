#include <cuda_runtime.h>
#include <stdio.h>

#include "utils.h"
void sumMatrix2d_CPU(float* A, float* B, float* C, const int nx, const int ny) {
  float* a = A;
  float* b = B;
  float* c = C;
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      c[i] = a[i] + b[i];
    }
    a += nx;
    b += nx;
    c += nx;
  }
}

__global__ void sumMatrix(float* A, float* B, float* C, int nx, int ny) {
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = ix + iy * nx;
  if (ix < nx && iy < ny) {
    C[idx] = A[idx] + A[idx];
  }
}
int main() {
  // set up device
  initDevice(0);

  int nx = 1 << 12;
  int ny = 1 << 12;
  int nxy = nx * ny;
  int nBytes = nxy * sizeof(float);

  // Malloc
  float* a_h = (float*)malloc(nBytes);
  float* b_h = (float*)malloc(nBytes);
  float* c_h = (float*)malloc(nBytes);
  float* c_from_gpu_h = (float*)malloc(nBytes);
  initialData(a_h, nxy);
  initialData(b_h, nxy);

  // cudaMalloc
  float* a_d = NULL;
  float* b_d = NULL;
  float* c_d = NULL;
  CHECK(cudaMalloc((void**)&a_d, nBytes));
  CHECK(cudaMalloc((void**)&b_d, nBytes));
  CHECK(cudaMalloc((void**)&c_d, nBytes));

  CHECK(cudaMemcpy(a_d, a_h, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(b_d, b_h, nBytes, cudaMemcpyHostToDevice));

  // cpu compute
  double iStart = cpuSecond();
  sumMatrix2d_CPU(a_h, b_h, c_h, nx, ny);
  double iElaps = cpuSecond() - iStart;
  printf("CPU Execution Time elapsed %f sec\n", iElaps);

  // gpu compute
  // 2d block and 2d grid
  int dimx = 32;
  int dimy = 32;
  dim3 block_0(dimx, dimy);
  dim3 grid_0((nx - 1) / block_0.x + 1, (ny - 1) / block_0.y + 1);
  iStart = cpuSecond();
  sumMatrix<<<grid_0, block_0>>>(a_d, b_d, c_d, nx, ny);
  CHECK(cudaDeviceSynchronize());
  iElaps = cpuSecond() - iStart;
  printf(
      "GPU Execution configuration<<<(%d,%d),(%d,%d)>>> Time elapsed %f sec\n",
      grid_0.x, grid_0.y, block_0.x, block_0.y, iElaps);
  CHECK(cudaMemcpy(c_from_gpu_h, c_d, nBytes, cudaMemcpyDeviceToHost));
  checkResult(c_from_gpu_h, c_h, nx);

  // 1d block and 1d grid
  dimx = 32;
  dim3 block_1(dimx);
  dim3 grid_1((nxy - 1) / block_1.x + 1);
  iStart = cpuSecond();
  sumMatrix<<<block_1, grid_1>>>(a_d, b_d, c_d, nx, ny);
  CHECK(cudaDeviceSynchronize());
  iElaps = cpuSecond() - iStart;
  printf(
      "GPU Execution configuration<<<(%d,%d),(%d,%d)>>> Time elapsed %f sec\n",
      grid_1.x, grid_1.y, block_1.x, block_1.y, iElaps);
  CHECK(cudaMemcpy(c_from_gpu_h, c_d, nBytes, cudaMemcpyDeviceToHost));
  checkResult(c_from_gpu_h, c_h, nx);

  // 1d block and 2d grid
  dimx = 32;
  dim3 block_2(dimx);
  dim3 grid_2((nx - 1) / block_1.x + 1, ny);
  iStart = cpuSecond();
  sumMatrix<<<block_2, grid_1>>>(a_d, b_d, c_d, nx, ny);
  CHECK(cudaDeviceSynchronize());
  iElaps = cpuSecond() - iStart;
  printf(
      "GPU Execution configuration<<<(%d,%d),(%d,%d)>>> Time elapsed %f sec\n",
      grid_2.x, grid_2.y, block_2.x, block_2.y, iElaps);
  CHECK(cudaMemcpy(c_from_gpu_h, c_d, nBytes, cudaMemcpyDeviceToHost));
  checkResult(c_from_gpu_h, c_h, nx);

  CHECK(cudaFree(a_d));
  CHECK(cudaFree(b_d));
  CHECK(cudaFree(c_d));
  free(a_h);
  free(b_h);
  free(c_h);
  free(c_from_gpu_h);
  return 0;
}