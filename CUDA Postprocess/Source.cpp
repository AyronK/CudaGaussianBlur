#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "CudaKernel.h"
#include <iostream>

using namespace std;
using namespace cv;

/// Global Variables
int DELAY_CAPTION = 1500;
int DELAY_BLUR = 10;
int MAX_KERNEL_LENGTH = 31;
int SIGMA_X = 250;
int SIGMA_Y = 250;

char window_name[] = "window";

/// Function headers
void TwoPassGaussianBlur();
int display_dst(int delay);

/**
* function main
*/
int main(int argc, char** argv)
{
	string filename;
	std::cout << "Write filename" << endl;
	std::cin >> filename;
	string width;
	std::cout << "Write width" << endl;
	std::cin >> width;
	string height;
	std::cout << "Write height" << endl;
	std::cin >> height;

	namedWindow(window_name, WINDOW_NORMAL);
	resizeWindow(window_name, std::stoi(width), std::stoi(height));
	
	IplImage* image;

	image = cvLoadImage(filename.c_str(), IMREAD_GRAYSCALE);
	if (!image)
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}


	IplImage* inputImage = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, image->nChannels);
	IplImage* outputImage = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, image->nChannels);

	cvConvert(image, inputImage);

	float *output = (float*)outputImage->imageData;
	float *input = (float*)inputImage->imageData;


	kernalGauss(input, output, image->width, image->height, inputImage->widthStep);

	cvScale(outputImage, outputImage, 1.0 / 255.0);

	cvShowImage(window_name, image);
	cvWaitKey(0);

	cvShowImage(window_name, outputImage);
	cvWaitKey(0);

	return 0;
}