#pragma once
#include <opencv2/opencv.hpp>

class DepthMapCalculator
{
public:
	cv::Mat compute(const cv::Mat& leftImage, const cv::Mat& rightImage, const int& minDisparity, const int& maxDisparity);

private:
	void computeGrayAndGradientImages(const cv::Mat& leftImage, const cv::Mat& rightImage);
	void computeLeftAndRightDisparityMap();
	void detectOcclusion();
	void convertDisparityMapToDepthMap();
	void fillOcclusionByWeightedMedianFilter();
	void computeDepthConfidence();
	//void fillLeftAndRightOcclusion();


	int _maxDisparity;
	int _minDisparity;
	cv::Mat _leftGrayImage;
	cv::Mat _rightGrayImage;
	cv::Mat _leftGradientImage;
	cv::Mat _rightGradientImage;

	cv::Mat _leftDisparityMap;
	cv::Mat _rightDisparityMap;
	cv::Mat _leftDepthMap8U;
//	cv::Mat _rightDisparityMap8U;

	cv::Mat _leftOcclusionMap;
//	cv::Mat _rightOcclusionMap;
};

