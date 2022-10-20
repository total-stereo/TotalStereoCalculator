#pragma once
#include <opencv2/opencv.hpp>


class StereoRectifier
{
public:
	std::tuple<cv::Mat, cv::Mat, int, int> compute(const cv::Mat& leftImage, const cv::Mat rightImage);

private:
	cv::Mat _descriptorLeftImage;
	cv::Mat _descriptorRightImage;

	std::vector<cv::KeyPoint> _keyPointsLeftImage;
	std::vector<cv::KeyPoint> _keyPointsRightImage;

	cv::Mat _currentTransformationMat = cv::Mat(cv::Matx23d{ 1,   0,  0,  0,  1,  0 });

	cv::Ptr<cv::SIFT> _siftOperator = cv::SIFT::create();
	cv::Ptr<cv::DescriptorMatcher> _matchingOperator =  cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
};