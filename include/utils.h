#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>

#define CHECK(call)                                                   \
  {                                                                   \
    const cudaError_t error = call;                                   \
    if (error != cudaSuccess) {                                       \
      printf("ERROR: %s:%d", __FILE__, __LINE__);                     \
      printf("code:%d, reason:%s", error, cudaGetErrorString(error)); \
      exit(1);                                                        \
    }                                                                 \
  }

void initialData(float *ip, int size) {
  time_t t;
  srand((unsigned)time(&t));
  for (int i = 0; i < size; i++) {
    ip[i] = int(rand() & 0xff);
  }
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
  double epsilon = 1.0E-8;
  for (int i = 0; i < N; i++) {
    if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
      printf("Results don\'t match!\n");
      printf("%f(hostRef[%d] )!= %f(gpuRef[%d])\n", hostRef[i], i, gpuRef[i],
             i);
      return;
    }
  }
  printf("Check result success!\n");
}

#endif  // UTILS_H