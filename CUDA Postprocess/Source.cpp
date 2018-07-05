#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "GaussKernel.h"
#include <iostream>

using namespace std;
using namespace cv;

/// Global Variables
int DELAY_CAPTION = 1500;
int DELAY_BLUR = 100;
int MAX_KERNEL_LENGTH = 31;
int SIGMA_X = 250;
int SIGMA_Y = 250;

char window_name[] = "window";
char window_name2[] = "window2";
char window_name3[] = "window3";

/**
* function main
*/
int main(int argc, char** argv)
{
	string filename = "lena.jpg";
	//std::cout << "Write filename" << endl;
	//std::cin >> filename;
	string width = "500";
	//std::cout << "Write width" << endl;
	//std::cin >> width;
	string height= "500";
	//std::cout << "Write height" << endl;
	//std::cin >> height;


	namedWindow(window_name, WINDOW_NORMAL);
	resizeWindow(window_name, std::stoi(width), std::stoi(height));

	namedWindow(window_name2, WINDOW_NORMAL);
	resizeWindow(window_name2, std::stoi(width), std::stoi(height));

	namedWindow(window_name3, WINDOW_NORMAL);
	resizeWindow(window_name3, std::stoi(width), std::stoi(height));
	
	IplImage* image;

	image = cvLoadImage(filename.c_str());
	if (!image)
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}


	IplImage* inputImage = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, image->nChannels);
	IplImage* outputImage = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, image->nChannels);
	IplImage* outputImage2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, image->nChannels);

	cvConvert(image, inputImage);

	float *output = (float*)outputImage->imageData;
	float *output2 = (float*)outputImage2->imageData;
	float *input = (float*)inputImage->imageData;

	int matrixSize = 3;

	while (matrixSize != 0) {
		std::cout << "Matrix size" << endl;
		std::cin >> matrixSize;

		// multiply by 3 if in rbg
		kernelGauss(input, output, image->width * 3, image->height, inputImage->widthStep, 1, matrixSize);
		kernelGauss(output, output2, image->width * 3, image->height, inputImage->widthStep, 0, matrixSize);

		cvScale(outputImage, outputImage, 1.0 / 255.0);
		cvScale(outputImage2, outputImage2, 1.0 / 255.0);

		cvShowImage(window_name, image);
		cvShowImage(window_name2, outputImage);
		cvShowImage(window_name3, outputImage2);

		cvWaitKey(0);
	}

	return 0;
}
