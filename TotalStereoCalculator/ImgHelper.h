#pragma once
#include <opencv2/opencv.hpp>

namespace ImgHelper
{
	cv::Mat convertUcharRGBToFloatGrayImage(const cv::Mat& rgbImage);

	void weightedMedianFilterSequentiell(const int& minDisparity, const int& maxDisparity, const int& k, const cv::Mat& grayValInputImage, cv::Mat& inOutImage);

}

