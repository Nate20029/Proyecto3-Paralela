/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : make
 ============================================================================
 */
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

//*****************************************************************
// The CPU function returns a pointer to the accummulator
void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
  *acc = new int[rBins * degreeBins];                // el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
  memset(*acc, 0, sizeof(int) * rBins * degreeBins); // init en ceros
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++)   // por cada pixel
    for (int j = 0; j < h; j++) //...
    {
      int idx = j * w + i;
      if (pic[idx] > 0) // si pasa thresh, entonces lo marca
      {
        int xCoord = i - xCent;
        int yCoord = yCent - j;                       // y-coord has to be reversed
        float theta = 0;                              // actual angle
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) // add 1 to all lines in that pixel
        {
          float r = xCoord * cos(theta) + yCoord * sin(theta);
          int rIdx = (r + rMax) / rScale;
          (*acc)[rIdx * degreeBins + tIdx]++; //+1 para este radio r y este theta
          theta += radInc;
        }
      }
    }
}

//*****************************************************************
// TODO usar memoria constante para la tabla de senos y cosenos
// inicializarlo en main y pasarlo al device
//__constant__ float d_Cos[degreeBins];
//__constant__ float d_Sin[degreeBins];

//*****************************************************************
// TODO Kernel memoria compartida
// __global__ void GPU_HoughTranShared(...)
// {
//   //TODO
// }
// TODO Kernel memoria Constante
// __global__ void GPU_HoughTranConst(...)
// {
//   //TODO
// }

// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
  // TODO calcular: int gloID = ?
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID >= w * h)
    return;

  int xCent = w / 2;
  int yCent = h / 2;

  // TODO explicar bien bien esta parte. Dibujar un rectangulo a modo de imagen sirve para visualizarlo mejor
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  // TODO eventualmente usar memoria compartida para el acumulador

  if (pic[gloID] > 0)
  {
    for (int tIdx = 0; tIdx < degreeBins; tIdx++)
    {
      // TODO utilizar memoria constante para senos y cosenos
      // float r = xCoord * cos(tIdx) + yCoord * sin(tIdx); //probar con esto para ver diferencia en tiempo
      float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
      int rIdx = (r + rMax) / rScale;
      // debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique
      atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
    }
  }

  // TODO eventualmente cuando se tenga memoria compartida, copiar del local al global
  // utilizar operaciones atomicas para seguridad
  // faltara sincronizar los hilos del bloque en algunos lados
}

//*****************************************************************
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


  float *pcCos = (float *)malloc(sizeof(float) * degreeBins);
  float *pcSin = (float *)malloc(sizeof(float) * degreeBins);
  float rad = 0;

  cudaMalloc((void **)&d_Cos, sizeof(float) * degreeBins);
  cudaMalloc((void **)&d_Sin, sizeof(float) * degreeBins);

  // CPU calculation
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);


  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos(rad);
    pcSin[i] = sin(rad);
    rad += radInc;
  }

  float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
  cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);

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

  // Launch the kernel
  int blockNum = ceil((float)w * h / 256.0);
  GPU_HoughTran<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale);

  // Record the stop event
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);

  // Calculate and print the elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU Hough Constante Transform tomo %f milisegundos\n", milliseconds);

  // Copy results back to host
  cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

  // Compare CPU and GPU results
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (abs(cpuht[i] != h_hough[i]) > tolerance)
      printf("Mismatch at index %d: CPU=%d, GPU=%d\n", i, cpuht[i], h_hough[i]);
  }

  // Free dynamically allocated memory
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
