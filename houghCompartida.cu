#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "common/pgm.h"

const int tolerance = 1;
const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
  *acc = new int[rBins * degreeBins];
  memset(*acc, 0, sizeof(int) * rBins * degreeBins);
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++)
    for (int j = 0; j < h; j++)
    {
      int idx = j * w + i;
      if (pic[idx] > 0)
      {
        int xCoord = i - xCent;
        int yCoord = yCent - j;
        float theta = 0;
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
          float r = xCoord * cos(theta) + yCoord * sin(theta);
          int rIdx = (r + rMax) / rScale;
          (*acc)[rIdx * degreeBins + tIdx]++;
          theta += radInc;
        }
      }
    }
}

__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID >= w * h)
    return;

  int xCent = w / 2;
  int yCent = h / 2;
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  if (pic[gloID] > 0)
  {
    for (int tIdx = 0; tIdx < degreeBins; tIdx++)
    {
      float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
      int rIdx = (r + rMax) / rScale;
      atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
    }
  }
}

__global__ void GPU_HoughTranShared(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID >= w * h)
    return;

  int xCent = w / 2;
  int yCent = h / 2;
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  extern __shared__ int localAcc[];

  for (int i = 0; i < degreeBins * rBins; i += blockDim.x)
    localAcc[threadIdx.x + i] = 0;

  __syncthreads();

  if (pic[gloID] > 0)
  {
    for (int tIdx = 0; tIdx < degreeBins; tIdx++)
    {
      float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
      int rIdx = (r + rMax) / rScale;
      atomicAdd(&localAcc[rIdx * degreeBins + tIdx], 1);
    }
  }

  __syncthreads();

  for (int i = 0; i < degreeBins * rBins; i += blockDim.x)
    atomicAdd(&acc[i + threadIdx.x], localAcc[i + threadIdx.x]);
}

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    printf("Usage: %s <image.pgm>\n", argv[0]);
    return -1;
  }

  int i;

  PGMImage inImg(argv[1]);

  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  float *d_Cos;
  float *d_Sin;

  cudaMalloc((void **)&d_Cos, sizeof(float) * degreeBins);
  cudaMalloc((void **)&d_Sin, sizeof(float) * degreeBins);

  CPU_HoughTran(inImg.pixels, w, h, &cpuht);

  float *pcCos = (float *)malloc(sizeof(float) * degreeBins);
  float *pcSin = (float *)malloc(sizeof(float) * degreeBins);
  float rad = 0;
  for (int i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos(rad);
    pcSin[i] = sin(rad);
    rad += radInc;
  }

  float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  cudaMemcpy(d_Cos, pcCos, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Sin, pcSin, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);

  unsigned char *d_in;
  int *d_hough;
  int *h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));

  cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
  cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
  cudaMemcpy(d_in, inImg.pixels, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

  // Define CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record the start event
  cudaEventRecord(start, NULL);

  // Launch the kernel with shared memory
  int blockNum = ceil((float)w * h / 256.0);
  int sharedMemorySize = degreeBins * rBins * sizeof(int);
  GPU_HoughTranShared<<<blockNum, 256, sharedMemorySize>>>(d_in, w, h, d_hough, rMax, rScale);

  // Record the stop event
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);

  // Calculate and print the elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU Hough Global, Constante y Compartida Transform took %f milliseconds\n", milliseconds);

  // Copy results back to host
  cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

  // Compare CPU and GPU results
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (abs(cpuht[i] != h_hough[i]) > tolerance)
      printf("Mismatch at index %d: CPU=%d, GPU=%d\n", i, cpuht[i], h_hough[i]);
  }

  // Free dynamically allocated memory
  cudaFree(d_Cos);
  cudaFree(d_Sin);
  cudaFree(d_in);
  cudaFree(d_hough);
  free(pcCos);
  free(pcSin);
  free(h_hough);
  delete[] cpuht;

  // Destroy the events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  printf("Done!\n");

  return 0;
}
