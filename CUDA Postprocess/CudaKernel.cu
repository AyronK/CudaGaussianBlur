#include<cuda.h>
#include<iostream>
#include <stdlib.h>
#include <stdio.h>
#include "CudaKernel.h"

using namespace std;

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
#define checkCudaErrors(err) __checkCudaErrors (err, __FILE__, __LINE__)


texture <float, 2, cudaReadModeElementType> tex1;

static cudaArray *cuArray = NULL;

//Kernel for x direction sobel
__global__ void gauss(float* output, int width, int height, int widthStep, float sigma)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float matrix[] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
	int s = matrix[0] + matrix[1] + matrix[2] + matrix[3] + matrix[4] + matrix[5] + matrix[6] + matrix[7] + matrix[8];

	if (x >= widthStep || y >= widthStep) {
		return;
	}

	float output_value = (matrix[0] * tex2D(tex1, x - sigma, y - sigma)) + (matrix[1] * tex2D(tex1, x, y - sigma)) + (matrix[2] * tex2D(tex1, x + sigma, y - sigma))
		+ (matrix[3] * tex2D(tex1, x - sigma, y)) + (matrix[4] * tex2D(tex1, x, y)) + (matrix[5] * tex2D(tex1, x + sigma, y))
		+ (matrix[6] * tex2D(tex1, x - sigma, y + sigma)) + (matrix[7] * tex2D(tex1, x, y + sigma)) + (matrix[8] * tex2D(tex1, x + sigma, y + sigma));

	output[y*widthStep + x] = output_value / s;
}


inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
	if (cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
			file, line, (int)err, cudaGetErrorString(err));
		exit(-1);
	}
}

//Host Code
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
inline void __cudaCheckError(const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		printf("cudaCheckError() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}

void kernelGauss(float* input, float* output, int width, int height, int widthStep, float sigma)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	CudaSafeCall(cudaMallocArray(&cuArray, &channelDesc, width, height));

	//Never use 1D memory copy if host and device pointers have different widthStep.
	// You don't know the width step of CUDA array, so its better to use cudaMemcpy2D...
	cudaMemcpy2DToArray(cuArray, 0, 0, input, widthStep, width * sizeof(float), height, cudaMemcpyHostToDevice);

	cudaBindTextureToArray(tex1, cuArray, channelDesc);

	float * D_output_x;
	CudaSafeCall(cudaMalloc(&D_output_x, widthStep*height));
	dim3 blocksize(16, 16);
	dim3 gridsize;
	gridsize.x = (width + blocksize.x - 1) / blocksize.x;
	gridsize.y = (height + blocksize.y - 1) / blocksize.y;

	gauss << < gridsize, blocksize >> > (D_output_x, width, height, widthStep / sizeof(float), sigma);
	cudaThreadSynchronize();
	CudaCheckError();

	//Don't forget to unbind the texture
	cudaUnbindTexture(tex1);

	CudaSafeCall(cudaMemcpy(output, D_output_x, height*widthStep, cudaMemcpyDeviceToHost));

	cudaFree(D_output_x);
	cudaFreeArray(cuArray);
}