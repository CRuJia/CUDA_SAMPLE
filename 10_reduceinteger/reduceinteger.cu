#include <cuda_runtime.h>
#include <stdio.h>
#include <utils.h>

int recursiveReduce(int *data, int const size) {
  if (size == 1) {
    return data[0];
  }
  int const stride = size / 2;

  for (int i = 0; i < stride; i++) {
    data[i] += data[i + stride];
  }
  return recursiveReduce(data, stride);
}

__global__ void warmup(int *g_idata, int *g_odata, int const n) {
  // set thread id
  unsigned int tid = threadIdx.x;
  // boundary check
  if (tid >= n) return;
  // convert global data pointer to local pointer
  int *idata = g_idata + blockIdx.x * blockDim.x;
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if (tid % (2 * stride) == 0) {
      idata[tid] += idata[tid + stride];
    }
    // synchronize threads
    __syncthreads();
  }
  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}

__global__ void reduceNeighbored(int *g_idata, int *g_odata, int const n) {
  // set thread id
  unsigned int tid = threadIdx.x;
  // covert global data pointer to local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x;
  // boundary check
  if (tid >= n) return;

  // in-place reduction in global memory
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if (tid % (2 * stride) == 0) {
      idata[tid] += idata[tid + stride];
    }
    // synchronize with in block
    __syncthreads();
  }

  // write result for this block to global memory
  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}

__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, int const n) {
  // set thread id
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // convert global data pointer to local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x;

  // boundary check
  if (idx >= n) return;

  // in-place reduction in global memory
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    // converet tid into local array index
    int index = 2 * stride * tid;
    if (index < blockDim.x) {
      idata[index] += idata[index + stride];
    }
    // synchronize with in block
    __syncthreads();
  }
  // write result for this block to global memory
  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}

__global__ void reduceInterleaved(int *g_idata, int *g_odata, int const n) {
  // set thread id
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // convert global data pointer to local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x;
  // boundary check
  if (idx >= n) return;

  // in-place reduction in global memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      idata[tid] += idata[tid + stride];
    }
    __syncthreads();
  }
  // write result for this block to global memory
  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}

__global__ void reduceUnrolling2(int *g_idata, int *g_odata, int const n) {
  // set thread id
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  // convert global data pointer to local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x * 2;

  // unrolling 2
  if (idx + blockDim.x < n) {
    g_idata[idx] += g_idata[idx + blockDim.x];
  }

  __syncthreads();

  // in-place reduction in global memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      idata[tid] += idata[tid + stride];
    }
    __syncthreads();
  }

  // write result for this block to global memory
  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}

__global__ void reduceUnrolling4(int *g_idata, int *g_odata, int const n) {
  // set thread id
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

  // convert global data pointer to local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x * 4;

  // unrolling 4
  if (idx + 3 * blockDim.x < n) {
    int a1 = g_idata[idx];
    int a2 = g_idata[idx + blockDim.x];
    int a3 = g_idata[idx + 2 * blockDim.x];
    int a4 = g_idata[idx + 3 * blockDim.x];
    g_idata[idx] = a1 + a2 + a3 + a4;
  }

  __syncthreads();

  // in-place reduction in global memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      idata[tid] += idata[tid + stride];
    }
    __syncthreads();
  }
  // write result for this block to global memory
  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}

__global__ void reduceUnrolling8(int *g_idata, int *g_odata, int const n) {
  // set thread id
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

  // convert global data pointer to local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x * 8;

  // unrolling 8
  if (idx + 7 * blockDim.x < n) {
    int a1 = g_idata[idx];
    int a2 = g_idata[idx + blockDim.x];
    int a3 = g_idata[idx + 2 * blockDim.x];
    int a4 = g_idata[idx + 3 * blockDim.x];
    int a5 = g_idata[idx + 4 * blockDim.x];
    int a6 = g_idata[idx + 5 * blockDim.x];
    int a7 = g_idata[idx + 6 * blockDim.x];
    int a8 = g_idata[idx + 7 * blockDim.x];
    g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
  }

  __syncthreads();

  // in-place reduction in global memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      idata[tid] += idata[tid + stride];
    }
    __syncthreads();
  }
  // write result for this block to global memory
  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}

__global__ void reduceUnrollWarps8(int *g_idata, int *g_odata, int const n) {
  // set thread id
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

  // convert global data pointer to local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x * 8;

  // unrolling 8
  if (idx + 7 * blockDim.x < n) {
    int a1 = g_idata[idx];
    int a2 = g_idata[idx + blockDim.x];
    int a3 = g_idata[idx + 2 * blockDim.x];
    int a4 = g_idata[idx + 3 * blockDim.x];
    int a5 = g_idata[idx + 4 * blockDim.x];
    int a6 = g_idata[idx + 5 * blockDim.x];
    int a7 = g_idata[idx + 6 * blockDim.x];
    int a8 = g_idata[idx + 7 * blockDim.x];
    g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
  }

  __syncthreads();

  // in-place reduction in global memory
  for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
      idata[tid] += idata[tid + stride];
    }
    __syncthreads();
  }

  // unrolling warp
  if (tid < 32) {
    volatile int *vmem = idata;
    vmem[tid] += vmem[tid + 32];
    vmem[tid] += vmem[tid + 16];
    vmem[tid] += vmem[tid + 8];
    vmem[tid] += vmem[tid + 4];
    vmem[tid] += vmem[tid + 2];
    vmem[tid] += vmem[tid + 1];
  }

  // write result for this block to global memory
  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}

__global__ void reduceCompleteUnrollWarps8(int *g_idata, int *g_odata,
                                           int const n) {
  // set thread id
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

  // convert global data pointer to local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x * 8;

  // unrolling 8
  if (idx + 7 * blockDim.x < n) {
    int a1 = g_idata[idx];
    int a2 = g_idata[idx + blockDim.x];
    int a3 = g_idata[idx + 2 * blockDim.x];
    int a4 = g_idata[idx + 3 * blockDim.x];
    int a5 = g_idata[idx + 4 * blockDim.x];
    int a6 = g_idata[idx + 5 * blockDim.x];
    int a7 = g_idata[idx + 6 * blockDim.x];
    int a8 = g_idata[idx + 7 * blockDim.x];
    g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
  }

  __syncthreads();

  // in-place reduction in global memory
  if (blockDim.x >= 1024 && tid < 512) {
    idata[tid] += idata[tid + 512];
  }
  __syncthreads();
  if (blockDim.x >= 512 && tid < 256) {
    idata[tid] += idata[tid + 256];
  }
  __syncthreads();
  if (blockDim.x >= 256 && tid < 128) {
    idata[tid] += idata[tid + 128];
  }
  __syncthreads();

  if (blockDim.x >= 128 && tid < 64) {
    idata[tid] += idata[tid + 64];
  }
  __syncthreads();

  // unrolling warp
  if (tid < 32) {
    volatile int *vmem = idata;
    vmem[tid] += vmem[tid + 32];
    vmem[tid] += vmem[tid + 16];
    vmem[tid] += vmem[tid + 8];
    vmem[tid] += vmem[tid + 4];
    vmem[tid] += vmem[tid + 2];
    vmem[tid] += vmem[tid + 1];
  }

  // write result for this block to global memory
  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}

int main(int argc, char **argv) {
  initDevice(0);

  bool bResult = false;

  int size = 1 << 24;
  size_t bytes = size * sizeof(int);

  printf("  with array size %d ", size);
  // execution configuration
  int blockSize = 1024;
  if (argc > 1) {
    blockSize = atoi(argv[1]);
  }
  dim3 block(blockSize, 1);
  dim3 grid((size + block.x - 1) / block.x, 1);
  printf(" grid %d block %d\n", grid.x, block.x);

  // allocate host memory
  int *idata_h = (int *)malloc(bytes);
  int *odata_h = (int *)malloc(grid.x * sizeof(int));
  int *tmp = (int *)malloc(bytes);

  // initialize the array
  initialData_int(idata_h, size);
  memcpy(tmp, idata_h, bytes);
  double iStart, iElaps;
  int gpu_sum = 0;

  // device memory
  int *idata_d = NULL;
  int *odata_d = NULL;
  cudaMalloc((void **)&idata_d, bytes);
  cudaMalloc((void **)&odata_d, grid.x * sizeof(int));

  // cpu reduction
  int cpu_sum = 0;
  iStart = cpuSecond();
  // cpu_sum = recursiveReduce(tmp, size);
  for (int i = 0; i < size; i++) {
    cpu_sum += tmp[i];
  }
  iElaps = cpuSecond() - iStart;
  printf("CPU reduction time: %.5f sec, result: %d\n", iElaps, cpu_sum);

  // kernel1: reduceNeighbored
  CHECK(cudaMemcpy(idata_d, idata_h, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  iStart = cpuSecond();
  warmup<<<grid, block>>>(idata_d, odata_d, size);
  CHECK(cudaDeviceSynchronize());
  iElaps = cpuSecond() - iStart;
  cudaMemcpy(odata_h, odata_d, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++) {
    gpu_sum += odata_h[i];
  }
  printf(
      "gpu warmup  elapsed %f sec gpu_sum: %d <<<grid %d block "
      "%d>>>\n",
      iElaps, gpu_sum, grid.x, block.x);

  // kerne1: reduceNeighbored
  CHECK(cudaMemcpy(idata_d, idata_h, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  iStart = cpuSecond();
  reduceNeighbored<<<grid, block>>>(idata_d, odata_d, size);
  CHECK(cudaDeviceSynchronize());
  iElaps = cpuSecond() - iStart;
  cudaMemcpy(odata_h, odata_d, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++) {
    gpu_sum += odata_h[i];
  }
  printf(
      "gpu reduceNeighbored  elapsed %f sec gpu_sum: %d <<<grid %d block "
      "%d>>>\n",
      iElaps, gpu_sum, grid.x, block.x);

  // kerne2: reduceNeighboredLess
  CHECK(cudaMemcpy(idata_d, idata_h, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  iStart = cpuSecond();
  reduceNeighboredLess<<<grid, block>>>(idata_d, odata_d, size);
  CHECK(cudaDeviceSynchronize());
  iElaps = cpuSecond() - iStart;
  cudaMemcpy(odata_h, odata_d, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++) {
    gpu_sum += odata_h[i];
  }
  printf(
      "gpu reduceNeighboredLess  elapsed %f sec gpu_sum: %d <<<grid %d block "
      "%d>>>\n",
      iElaps, gpu_sum, grid.x, block.x);

  // kerne3: reduceInterleaved
  CHECK(cudaMemcpy(idata_d, idata_h, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  iStart = cpuSecond();
  reduceInterleaved<<<grid, block>>>(idata_d, odata_d, size);
  CHECK(cudaDeviceSynchronize());
  iElaps = cpuSecond() - iStart;
  cudaMemcpy(odata_h, odata_d, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++) {
    gpu_sum += odata_h[i];
  }
  printf(
      "gpu reduceInterleaved  elapsed %f sec gpu_sum: %d <<<grid %d block "
      "%d>>>\n",
      iElaps, gpu_sum, grid.x, block.x);

  // kerne4: reduceUnrolling2
  CHECK(cudaMemcpy(idata_d, idata_h, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  iStart = cpuSecond();
  reduceUnrolling2<<<grid.x / 2, block>>>(idata_d, odata_d, size);
  CHECK(cudaDeviceSynchronize());
  iElaps = cpuSecond() - iStart;
  cudaMemcpy(odata_h, odata_d, grid.x / 2 * sizeof(int),
             cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x / 2; i++) {
    gpu_sum += odata_h[i];
  }
  printf(
      "gpu reduceUnrolling2  elapsed %f sec gpu_sum: %d <<<grid %d block "
      "%d>>>\n",
      iElaps, gpu_sum, grid.x / 2, block.x);

  // kerne5: reduceUnrolling4
  CHECK(cudaMemcpy(idata_d, idata_h, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  iStart = cpuSecond();
  reduceUnrolling4<<<grid.x / 4, block>>>(idata_d, odata_d, size);
  CHECK(cudaDeviceSynchronize());
  iElaps = cpuSecond() - iStart;
  cudaMemcpy(odata_h, odata_d, grid.x / 4 * sizeof(int),
             cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x / 4; i++) {
    gpu_sum += odata_h[i];
  }
  printf(
      "gpu reduceUnrolling4  elapsed %f sec gpu_sum: %d <<<grid %d block "
      "%d>>>\n",
      iElaps, gpu_sum, grid.x / 4, block.x);

  // kerne6: reduceUnrolling8
  CHECK(cudaMemcpy(idata_d, idata_h, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  iStart = cpuSecond();
  reduceUnrolling8<<<grid.x / 8, block>>>(idata_d, odata_d, size);
  CHECK(cudaDeviceSynchronize());
  iElaps = cpuSecond() - iStart;
  cudaMemcpy(odata_h, odata_d, grid.x / 8 * sizeof(int),
             cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x / 8; i++) {
    gpu_sum += odata_h[i];
  }
  printf(
      "gpu reduceUnrolling8  elapsed %f sec gpu_sum: %d <<<grid %d block "
      "%d>>>\n",
      iElaps, gpu_sum, grid.x / 8, block.x);

  // kerne7: reduceUnrollWarps8
  CHECK(cudaMemcpy(idata_d, idata_h, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  iStart = cpuSecond();
  reduceUnrollWarps8<<<grid.x / 8, block>>>(idata_d, odata_d, size);
  CHECK(cudaDeviceSynchronize());
  iElaps = cpuSecond() - iStart;
  cudaMemcpy(odata_h, odata_d, grid.x / 8 * sizeof(int),
             cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x / 8; i++) {
    gpu_sum += odata_h[i];
  }
  printf(
      "gpu reduceUnrollWarps8  elapsed %f sec gpu_sum: %d <<<grid %d block "
      "%d>>>\n",
      iElaps, gpu_sum, grid.x / 8, block.x);

  // kerne8: reduceCompleteUnrollWarps8
  CHECK(cudaMemcpy(idata_d, idata_h, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  iStart = cpuSecond();
  reduceCompleteUnrollWarps8<<<grid.x / 8, block>>>(idata_d, odata_d, size);
  CHECK(cudaDeviceSynchronize());
  iElaps = cpuSecond() - iStart;
  cudaMemcpy(odata_h, odata_d, grid.x / 8 * sizeof(int),
             cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x / 8; i++) {
    gpu_sum += odata_h[i];
  }
  printf(
      "gpu reduceCompleteUnrollWarps8  elapsed %f sec gpu_sum: %d <<<grid %d "
      "block "
      "%d>>>\n",
      iElaps, gpu_sum, grid.x / 8, block.x);

  // free cpu memory
  free(tmp);
  free(idata_h);
  free(odata_h);
  CHECK(cudaFree(idata_d));
  CHECK(cudaFree(odata_d));

  // reset device
  CHECK(cudaDeviceReset());

  // check result
  bResult = (cpu_sum == gpu_sum);
  printf("Test %s\n", bResult ? "Passed" : "Failed");

  return 0;
}