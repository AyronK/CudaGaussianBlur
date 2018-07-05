#include<cuda.h>
#include<iostream>
#include <stdlib.h>
#include <stdio.h>
#include "GaussKernel.h"
using namespace std;

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )

texture <float, 2, cudaReadModeElementType> tex1;

static cudaArray *cuArray = NULL;

#define VERTICAL 0
#define HORIZONTAL 1

__global__ void gauss(float* output, int width, int height, int widthStep, float sigma, int direction, int matrixSize, int* matrix)
{	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;


	int m3[] = { 1,2,1 };
	int m5[] = {1,4,6,4,1};
	int m7[] = { 2,22,97,159,97,22,2};
	if (matrixSize == 3)
		matrix = m3;
	else if (matrixSize == 5)
		matrix = m5;
	else
		matrix = m7;
	int s = 0;

	for (int i = 0; i < matrixSize; i++)
	{
		s += matrix[i];
	}

	if (x >= widthStep || y >= widthStep) {
		return;
	}

	float outputValue = 0;

	if (direction == VERTICAL) {
		for (int j = 0; j < matrixSize; j++) {
			int x_offset = j - matrixSize / 2;
			outputValue += matrix[j] * tex2D(tex1, x + x_offset * sigma * 3, y);
		}
	}
	else if (direction == HORIZONTAL) {
		for (int j = 0; j < matrixSize; j++) {
			int x_offset = j - matrixSize / 2;
			outputValue += matrix[j] * tex2D(tex1, x , y + x_offset * sigma);
		}
		}

	output[y*widthStep + x] = outputValue/s;
}

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	if (cudaSuccess != err)
	{
		printf("cudaSafeCall() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}

void kernelGauss(float* input, float* output, int width, int height, int widthStep, int direction, int matrixSize)
{
	int* matrix = (int*)malloc(sizeof(int)*matrixSize*matrixSize);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	CudaSafeCall(cudaMallocArray(&cuArray, &channelDesc, width, height));

	cudaMemcpy2DToArray(cuArray, 0, 0, input, widthStep, width * sizeof(float), height, cudaMemcpyHostToDevice);

	cudaBindTextureToArray(tex1, cuArray, channelDesc);

	float * D_output_x;
	int *D_matrix;
	CudaSafeCall(cudaMalloc(&D_output_x, widthStep*height));
	CudaSafeCall(cudaMalloc(&D_matrix, matrixSize*sizeof(int)));
	//cudaMallocManaged(&D_matrix, sizeof(matrix), cudaMemcpyHostToDevice);
	//memcpy(D_matrix, matrix, matrixSize*matrixSize*sizeof(int));
	dim3 blocksize(16, 16);
	dim3 gridsize;
	gridsize.x = (width + blocksize.x - 1) / blocksize.x;
	gridsize.y = (height + blocksize.y - 1) / blocksize.y;
	gauss << < gridsize, blocksize >> > (D_output_x, width, height, widthStep / sizeof(float), 10, direction, matrixSize,D_matrix);
	cudaThreadSynchronize();

	cudaUnbindTexture(tex1);

	CudaSafeCall(cudaMemcpy(output, D_output_x, height*widthStep, cudaMemcpyDeviceToHost));

	cudaFree(D_output_x);
	cudaFree(D_matrix);
	cudaFree(matrix);
	cudaFreeArray(cuArray);
	delete[] matrix;
}