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
	//int* matrix(new int[matrixSize*matrixSize]);  //{ 1, 2, 1, 2, 4, 2, 1, 2, 1 };
	//int matrix[9] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
	/*matrix[0] = 1;

	for (int i = 1; i < matrixSize; i++)
	{
		if (i <= matrixSize / 2)
			matrix[i] = matrix[i - 1] * 2;
		else
			matrix[i] = matrix[i - 1] / 2;
	}

	for (int i = 1; i < matrixSize; i++) {
		for (int j = 0; j < matrixSize; j++) {
			if (i <= matrixSize / 2) {
				matrix[i*matrixSize + j] = matrix[((i-1)*matrixSize) + j] * 2;
			}
			else {
				matrix[i*matrixSize + j] = matrix[((i-1)*matrixSize) + j] / 2;
			}
		}
	}
*/
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	/*if (direction == VERTICAL) {
		matrix[3] = 0;
		matrix[4] = 0;
		matrix[5] = 0;
	}
	else {
		matrix[1] = 0;
		matrix[4] = 0;
		matrix[7] = 0;
	}*/

	int s = 0;

	for (int i = 0; i < matrixSize*matrixSize; i++)
	{
		s += matrix[i];
	}

	if (x >= widthStep || y >= widthStep) {
		return;
	}

	float outputValue = (matrix[0] * tex2D(tex1, x - sigma, y - sigma)) + (matrix[1] * tex2D(tex1, x, y - sigma)) + (matrix[2] * tex2D(tex1, x + sigma, y - sigma))
			+ (matrix[3] * tex2D(tex1, x - sigma, y)) + (matrix[4] * tex2D(tex1, x, y)) + (matrix[5] * tex2D(tex1, x + sigma, y))
			+ (matrix[6] * tex2D(tex1, x - sigma, y + sigma)) + (matrix[7] * tex2D(tex1, x, y + sigma)) + (matrix[8] * tex2D(tex1, x + sigma, y + sigma));
		
	output[y*widthStep + x] = outputValue / 16;
	delete[] matrix;
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

void kernelGauss(float* input, float* output, int width, int height, int widthStep, float sigma, int direction, int matrixSize)
{
	int* matrix = (int*)malloc(sizeof(int)*matrixSize*matrixSize);
	matrix[0] = 1;

	for (int i = 1; i < matrixSize; i++)
	{
		if (i <= matrixSize / 2)
			matrix[i] = matrix[i - 1] * 2;
		else
			matrix[i] = matrix[i - 1] / 2;
	}

	for (int i = 1; i < matrixSize; i++) {
		for (int j = 0; j < matrixSize; j++) {
			if (i <= matrixSize / 2) {
				matrix[i*matrixSize + j] = matrix[((i - 1)*matrixSize) + j] * 2;
			}
			else {
				matrix[i*matrixSize + j] = matrix[((i - 1)*matrixSize) + j] / 2;
			}
		}
	}

	for (int i = 0; i < matrixSize; i++)
	{
		for (int j = 0; j < matrixSize; j++)
		{
			cout << matrix[i*matrixSize + j] << "\t";
		}
		cout << endl;
	}

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	CudaSafeCall(cudaMallocArray(&cuArray, &channelDesc, width, height));

	cudaMemcpy2DToArray(cuArray, 0, 0, input, widthStep, width * sizeof(float), height, cudaMemcpyHostToDevice);

	cudaBindTextureToArray(tex1, cuArray, channelDesc);

	float * D_output_x;
	int *D_matrix;
	CudaSafeCall(cudaMalloc(&D_output_x, widthStep*height));
	cudaMallocManaged(&D_matrix, sizeof(matrix));
	memcpy(D_matrix, matrix, matrixSize*matrixSize*sizeof(int));
	dim3 blocksize(16, 16);
	dim3 gridsize;
	gridsize.x = (width + blocksize.x - 1) / blocksize.x;
	gridsize.y = (height + blocksize.y - 1) / blocksize.y;
	gauss << < gridsize, blocksize >> > (D_output_x, width, height, widthStep / sizeof(float), sigma, direction, matrixSize,D_matrix);
	cudaThreadSynchronize();

	cudaUnbindTexture(tex1);

	CudaSafeCall(cudaMemcpy(output, D_output_x, height*widthStep, cudaMemcpyDeviceToHost));

	cudaFree(D_output_x);
	cudaFree(matrix);
	cudaFreeArray(cuArray);
	//delete[] matrix;
}