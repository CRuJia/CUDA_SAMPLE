#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
__global__ void warmup(float* c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a=0.0;
    float b = 0.0;
    if((tid/warpSize)%2==0){
        a = 100.f;
    }else{
        b = 200.f;
    }
    c[tid] = a+b;
}

__global__ void mathKernel1(float* c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a=0.0;
    float b = 0.0;
    if(tid%2==0){
        a = 100.f;
    }else{
        b = 200.f;
    }
    c[tid] = a+b;
}

__global__ void mathKernel2(float* c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a=0.0;
    float b = 0.0;
    if((tid/warpSize)%2==0){
        a = 100.f;
    }else{
        b = 200.f;
    }
    c[tid] = a+b;
}

__global__ void mathKernel3(float* c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a=0.0;
    float b = 0.0;
    bool ipred = (tid%2==0);
    if(ipred){
        a = 100.f;
    }else{
        b = 200.f;
    }
    c[tid] = a+b;
}

int main(int argc, char **argv) { 
    initDevice(0);

    //setup data size
    int size=64;
    int blocksize=64;
    if(argc>1) blocksize = atoi(argv[1]);
    if(argc>2) size=atoi(argv[2]);
    size_t nBytes = size * sizeof(float);
    printf("Data size %d", size);

    // setup execution configuration
    dim3 block(blocksize, 1);
    dim3 grid((size-1)/1+1,1);
    printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

    // malloc cpu memory
    float* C_h = (float*)malloc(nBytes);
    //allocate gpu memory
    float* C_d = NULL;
    CHECK(cudaMalloc((float**)&C_d,nBytes));

    //run a warmup kernel to remove overhead
	double iStart, iElaps;
	cudaDeviceSynchronize();
    iStart = cpuSecond();
    warmup<<<grid,block>>> (C_d);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
    printf("warmup	  <<<%d,%d>>>elapsed %lf sec \n", grid.x, block.x, iElaps);

    // run kernel 1
    iStart = cpuSecond();
    mathKernel1<<<grid,block>>> (C_d);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
    printf("mathKernel1 <<<%d,%d>>>elapsed %lf sec \n", grid.x, block.x, iElaps);
    cudaMemcpy(C_h, C_d, nBytes, cudaMemcpyDeviceToHost);

    //run kernel 2
    iStart = cpuSecond();
    mathKernel2<<<grid,block>>> (C_d);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
    printf("mathKernel2 <<<%d,%d>>>elapsed %lf sec \n", grid.x, block.x, iElaps);
    cudaMemcpy(C_h, C_d, nBytes, cudaMemcpyDeviceToHost);

    //run kernel 3
    iStart = cpuSecond();
    mathKernel3<<<grid,block>>> (C_d);     
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("mathKernel3 <<<%d,%d>>>elapsed %lf sec \n", grid.x, block.x, iElaps);
    cudaMemcpy(C_h, C_d, nBytes, cudaMemcpyDeviceToHost);

    // free gpu memory
    CHECK(cudaFree(C_d));
    // free cpu memory
    free(C_h);

    cudaDeviceReset();
    return 0; }