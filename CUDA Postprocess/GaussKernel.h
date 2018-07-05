#pragma once
#include "cuda.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

void kernelGauss(float* input, float* output, int width, int height, int widthStep, float sigma, int direction, int matrixSize);