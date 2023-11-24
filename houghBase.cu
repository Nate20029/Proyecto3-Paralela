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

__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin)
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

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    printf("Usage: %s <image.pgm>\n", argv[0]);
    return -1;
  }

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

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);

  int blockNum = ceil((float)w * h / 256.0);
  GPU_HoughTran<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);

  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU Hough Transform took %f milliseconds\n", milliseconds);

  cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

  for (int i = 0; i < degreeBins * rBins; i++)
  {
    if (abs(cpuht[i] != h_hough[i]) > tolerance)
      printf("Mismatch at index %d: CPU=%d, GPU=%d\n", i, cpuht[i], h_hough[i]);
  }

  cudaFree(d_Cos);
  cudaFree(d_Sin);
  cudaFree(d_in);
  cudaFree(d_hough);
  free(pcCos);
  free(pcSin);
  free(h_hough);
  delete[] cpuht;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  printf("Done!\n");

  return 0;
}
