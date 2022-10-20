#include "DepthMapCalculator.h"
#include "ImgHelper.h"

#include <omp.h>
#include <opencv2/ximgproc/edge_filter.hpp>


float computeCostColor(const float& color1, const float& color2, const float& maxCost)
{
    return std::min(std::abs(color1 - color2), maxCost);
}

float computeCostGradient(const float& gradient1, const float& gradient2, const float& maxCost)
{
    float cost = std::abs(gradient1 - gradient2);
    return std::min(cost, maxCost);
}

cv::Mat DepthMapCalculator::compute(const cv::Mat& leftImage, const cv::Mat& rightImage, const int& minDisparity, const int& maxDisparity)
{
    _maxDisparity = maxDisparity;
    _minDisparity = minDisparity;

    computeGrayAndGradientImages(leftImage, rightImage);

    computeLeftAndRightDisparityMap();

    detectOcclusion();

    convertDisparityMapToDepthMap();

    fillOcclusionByWeightedMedianFilter();

    fillOcclusionByWeightedMedianFilter();

    return _leftDepthMap8U;
}

void DepthMapCalculator::computeGrayAndGradientImages(const cv::Mat& leftImage, const cv::Mat& rightImage)
{
    _leftGrayImage = ImgHelper::convertUcharRGBToFloatGrayImage(leftImage);
    _rightGrayImage = ImgHelper::convertUcharRGBToFloatGrayImage(rightImage);

    cv::Sobel(_leftGrayImage, _leftGradientImage, CV_32F, 1, 0, 5, 1.0, 0.0, cv::BORDER_REFLECT);
    cv::Sobel(_rightGrayImage, _rightGradientImage, CV_32F, 1, 0, 5, 1.0, 0.0, cv::BORDER_REFLECT);

   /* cv::Mat yDirLeft, yDirRight;
    cv::Sobel(_leftGrayImage, yDirLeft, CV_32F, 0, 1, 5, 1.0, 0.0, cv::BORDER_REFLECT);
    cv::Sobel(_rightGrayImage, yDirRight, CV_32F, 0, 1, 5, 1.0, 0.0, cv::BORDER_REFLECT);
    _leftGradientImage = abs(_leftGradientImage * 0.5) + abs(yDirLeft * 0.5);
    _rightGradientImage = abs(_rightGradientImage * 0.5) + abs(yDirRight * 0.5);*/
}

void DepthMapCalculator::computeLeftAndRightDisparityMap()
{
    int filterRadius = 16; // 27;// 25;
    float maxCostColor = 30 / 255.f / 3.0;; // 60.f / 255.f / 3.0;
    float maxCostGradient = 10 / 255.f; // / 50.f / 255.f;
    float alpha = 0.9f;
    float epsilon = 0.000025f; // 0.0025f; // 25f;

    cv::Mat minCostMapLeft(_leftGrayImage.size(), CV_32FC1, cv::Scalar(std::numeric_limits<float>::max()));
    cv::Mat minCostMapRight(_rightGrayImage.size(), CV_32FC1, cv::Scalar(std::numeric_limits<float>::max()));

    cv::Mat costMapLeft(_leftGrayImage.size(), CV_32FC1);
    cv::Mat costMapRight(_leftGrayImage.size(), CV_32FC1);

    _leftDisparityMap = cv::Mat(_leftGrayImage.size(), CV_32SC1, cv::Scalar(_minDisparity -1));
    _rightDisparityMap = cv::Mat(_leftGrayImage.size(), CV_32SC1, cv::Scalar(_minDisparity -1));

    auto filterLeft = cv::ximgproc::createGuidedFilter(_rightGrayImage, filterRadius, epsilon);
    auto filterRight = cv::ximgproc::createGuidedFilter(_rightGrayImage, filterRadius, epsilon);

    float maxCostVal = (1.f - alpha) * maxCostColor + alpha * maxCostGradient;

    // 1. calculate disparity Map
    for (int d = _minDisparity; d <= _maxDisparity; d++)
    {
        //right = left-d;
        //d  = left-right

        int xStart = 0;
        int xStop = _leftDisparityMap.cols;
        if (d > 0)
        {
            costMapLeft(cv::Rect(0, 0, d, costMapLeft.rows)) = maxCostVal;
            costMapRight(cv::Rect(costMapRight.cols - d , 0, d, costMapRight.rows)) = maxCostVal;
            xStart = d;
        }
        else if(d < 0)
        {
            costMapLeft(cv::Rect(costMapRight.cols + d , 0, -d, costMapRight.rows)) = maxCostVal;
            costMapRight(cv::Rect(0, 0, -d, costMapLeft.rows)) = maxCostVal;
            xStop += d;
        }


        //compute cost for left and right image
#pragma omp parallel for
        for (int yIdx = 0; yIdx < _leftDisparityMap.rows; yIdx++)
        {
            float* dataCostLeft = costMapLeft.ptr<float>(yIdx);
            float* dataCostRight = costMapRight.ptr<float>(yIdx);

            for (int xIdxLeft = xStart; xIdxLeft < xStop; xIdxLeft++)
            {
                int xIdxRight = xIdxLeft - d; //xIdx2 == rechts
                float costColor = computeCostColor(_leftGrayImage.at<float>(yIdx, xIdxLeft), _rightGrayImage.at<float>(yIdx, xIdxRight), maxCostColor);
                float costGradient = computeCostGradient(_leftGradientImage.at<float>(yIdx, xIdxLeft), _rightGradientImage.at<float>(yIdx, xIdxRight), maxCostGradient);
                
                float costValue = (1.f - alpha) * costColor + alpha * costGradient;
                dataCostLeft[xIdxLeft] = costValue;
                dataCostRight[xIdxRight] = costValue;
            }
        }

        
        //filter Cost
        cv::Mat filteredCostImageLeft, filteredCostImageRight;
        filterLeft->filter(costMapLeft, filteredCostImageLeft);
        filterRight->filter(costMapRight, filteredCostImageRight);

        //merge -> the winner takes it all
#pragma omp parallel for
        for (int yIdx = 0; yIdx < _leftDisparityMap.rows; yIdx++)
        {
            for (int xIdx = 0; xIdx < _leftDisparityMap.cols; xIdx++)
            {
                if (filteredCostImageLeft.at<float>(yIdx, xIdx) < minCostMapLeft.at<float>(yIdx, xIdx))
                {
                    minCostMapLeft.at<float>(yIdx, xIdx) = filteredCostImageLeft.at<float>(yIdx, xIdx);
                    _leftDisparityMap.at<int>(yIdx, xIdx) = d;
                }

                if (filteredCostImageRight.at<float>(yIdx, xIdx) < minCostMapRight.at<float>(yIdx, xIdx))
                {
                    minCostMapRight.at<float>(yIdx, xIdx) = filteredCostImageRight.at<float>(yIdx, xIdx);
                    _rightDisparityMap.at<int>(yIdx, xIdx) = d;
                }
            }
        }
    }

}

void DepthMapCalculator::detectOcclusion()
{
    int maxDiffDisparity = 1;
    _leftOcclusionMap = cv::Mat::zeros(_leftDisparityMap.size(), CV_8UC1);

    //compute occlusion
#pragma omp parallel for
    for (int yIdx = 0; yIdx < _leftOcclusionMap.rows; yIdx++)
    {
        for (int xIdxLeft = 0; xIdxLeft < _leftOcclusionMap.cols; xIdxLeft++)
        {
            int dLeft = _leftDisparityMap.at<int>(yIdx, xIdxLeft);

            int xIdxRight = xIdxLeft - dLeft;

            if (xIdxRight >= 0 && xIdxRight < _rightDisparityMap.cols)
            {
                int dRight = _rightDisparityMap.at<int>(yIdx, xIdxRight);

                int diffDisparity = std::abs(dLeft - dRight);
                if (diffDisparity <= maxDiffDisparity)
                {
                    _leftOcclusionMap.at<uchar>(yIdx, xIdxLeft) = 255;
                }
            }
        }
    }


    cv::Mat validRange;
    cv::inRange(_leftDisparityMap, cv::Scalar(_minDisparity), cv::Scalar(_maxDisparity), validRange);
    cv::bitwise_and(_leftOcclusionMap, validRange, _leftOcclusionMap);
}

void DepthMapCalculator::convertDisparityMapToDepthMap()
{
    cv::multiply(_leftDisparityMap, cv::Scalar(-1), _leftDisparityMap);
    cv::normalize(_leftDisparityMap, _leftDepthMap8U, 255, 0, cv::NORM_MINMAX, CV_8U);

    //normally :depth ~ 1/disparity
    //shift min Disparity Val to value 1
  /*  double scaleFactor = 255.0 / _maxDisparity;
    _leftDisparityMap.convertTo(_leftDepthMap8U, CV_8UC1, -scaleFactor, 255.0 + _minDisparity * scaleFactor +1 );*/

/*    float dispMaxShifted = _maxDisparity - _minDisparity + 1.f;
    float minDepthVal = 1.f / dispMaxShifted;

#pragma omp parallel for
    for (int yIdx = 0; yIdx < _leftDisparityMap.rows; yIdx++)
    {
        uchar* dataLeftDepthMap = _leftDepthMap8U.ptr<uchar>(yIdx);
        const int* dataLeftDispMap = _leftDisparityMap.ptr<int>(yIdx);

        for (int xIdx = 0; xIdx < _leftDisparityMap.cols; xIdx++)
        {
            float shiftedDisp = static_cast<float>(dataLeftDispMap[xIdx] -_minDisparity) + 1.f;
            dataLeftDepthMap[xIdx] = static_cast<uchar>(255.f * (1.f / shiftedDisp - minDepthVal));
        }
    }

    cv::Mat a = _leftDepthMap8U;*/
    cv::Mat a = _leftDepthMap8U;
    cv::bitwise_and(_leftDepthMap8U, _leftOcclusionMap, _leftDepthMap8U); //

}

void DepthMapCalculator::fillOcclusionByWeightedMedianFilter()
{
    int k = 21;
    int radius = k * 0.5;

    cv::Mat depthImageWithBorder, grayValWithBorder;
    cv::copyMakeBorder(_leftDepthMap8U, depthImageWithBorder, radius, radius, radius, radius, cv::BORDER_REFLECT101);
    cv::copyMakeBorder(_leftGrayImage, grayValWithBorder, radius, radius, radius, radius, cv::BORDER_REFLECT101);

    cv::Mat outputImageTopLeft = depthImageWithBorder.clone();
    ImgHelper::weightedMedianFilterSequentiell(0, 255, k, grayValWithBorder, outputImageTopLeft);

    cv::Mat outputImageBottomRight = depthImageWithBorder.clone();
    cv::flip(outputImageBottomRight, outputImageBottomRight, -1);
    ImgHelper::weightedMedianFilterSequentiell(0, 255, k, grayValWithBorder, outputImageBottomRight);
    cv::flip(outputImageBottomRight, outputImageBottomRight, -1);

    cv::Mat outputImageTopRight = depthImageWithBorder.clone();
    cv::flip(outputImageTopRight, outputImageTopRight, 0);
    ImgHelper::weightedMedianFilterSequentiell(0, 255, k, grayValWithBorder, outputImageTopRight);
    cv::flip(outputImageTopRight, outputImageTopRight, 0);

    cv::Mat outputImageBottomLeft = depthImageWithBorder.clone();
    cv::flip(outputImageBottomLeft, outputImageBottomLeft, 1);
    ImgHelper::weightedMedianFilterSequentiell(0, 255, k, grayValWithBorder, outputImageBottomLeft);
    cv::flip(outputImageBottomLeft, outputImageBottomLeft, 1);

    cv::Mat outputImage = outputImageTopLeft;
    cv::max(outputImage, outputImageBottomRight, outputImage);
    cv::max(outputImage, outputImageTopRight, outputImage);
    cv::max(outputImage, outputImageBottomLeft, outputImage);

    _leftDepthMap8U = outputImage(cv::Rect(radius, radius, _leftDepthMap8U.cols, _leftDepthMap8U.rows));
}

