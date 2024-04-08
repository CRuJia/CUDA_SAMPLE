#include <stdio.h>

__global__ void cuda_hello() { printf("Hello World from GPU!\n"); }

int main(int arg, char **argv) {
  printf("CPU: Hello World!\n");
  cuda_hello<<<1, 10>>>();
  cudaDeviceReset();
  return 0;
}