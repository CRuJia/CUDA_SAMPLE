#include <cuda_runtime.h>
#include <stdio.h>

#include "utils.h"

void sumMatrix2D_CPU(float* MatA, float* MatB, float* MatC, int nx, int ny) {
  float* a = MatA;
  float* b = MatB;
  float* c = MatC;
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      c[i] = a[i] + b[i];
    }
    a += nx;
    b += nx;
    c += nx;
  }
}

__global__ void sumMatrix(float* MatA, float* MatB, float* MatC, int nx,
                          int ny) {
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = iy * nx + ix;
  if (ix < nx && iy < ny) {
    MatC[idx] = MatA[idx] + MatB[idx];
  }
}

int main(int argc, char** argv) {
  initDevice(0);

  int nx = 1 << 12;
  int ny = 1 << 12;
  int nxy = nx * ny;
  int nBytes = nxy * sizeof(float);

  // Allocate host memory
  float* A_h = (float*)malloc(nBytes);
  float* B_h = (float*)malloc(nBytes);
  float* C_h = (float*)malloc(nBytes);
  float* C_from_gpu = (float*)malloc(nBytes);

  // Initialize host memory
  initialData(A_h, nxy);
  initialData(B_h, nxy);

  // cuda malloc
  float* A_d = NULL;
  float* B_d = NULL;
  float* C_d = NULL;
  CHECK(cudaMalloc((void**)&A_d, nBytes));
  CHECK(cudaMalloc((void**)&B_d, nBytes));
  CHECK(cudaMalloc((void**)&C_d, nBytes));

  // Copy host memory to device
  CHECK(cudaMemcpy(A_d, A_h, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(B_d, B_h, nBytes, cudaMemcpyHostToDevice));

  // cpu computation
  double istart = cpuSecond();
  sumMatrix2D_CPU(A_h, B_h, C_h, nx, ny);
  double iElaps = cpuSecond() - istart;
  printf("CPU sumMatrix2D elapsed time: %f(s)\n", iElaps);

  // gpu computation
  int dimx = 32;
  int dimy = 32;
  if (argc > 1) dimx = atoi(argv[1]);
  if (argc > 2) dimy = atoi(argv[2]);

  dim3 block(dimx, dimy);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
  istart = cpuSecond();
  sumMatrix<<<grid, block>>>(A_d, B_d, C_d, nx, ny);
  CHECK(cudaDeviceSynchronize());
  iElaps = cpuSecond() - istart;
  printf(
      "GPU EXecution configuration:<<<(%d, %d), (%d, %d)>>> Time elapsed: "
      "%f(s)\n",
      grid.x, grid.y, block.x, block.y, iElaps);
  CHECK(cudaMemcpy(C_from_gpu, C_d, nBytes, cudaMemcpyDeviceToHost));

  // Check result
  checkResult(C_h, C_from_gpu, nxy);

  // Free device memory
  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B_d));
  CHECK(cudaFree(C_d));

  // Free host memory
  free(A_h);
  free(B_h);
  free(C_h);
  free(C_from_gpu);

  // Reset device
  CHECK(cudaDeviceReset());
  return 0;
}
