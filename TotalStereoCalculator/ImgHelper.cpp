#include "ImgHelper.h"
#include <omp.h>

float cumSum(std::vector<float>& histogram)
{
    for (int i = 1; i < histogram.size(); i++)
    {
        histogram[i] += histogram[i - 1];
    }

    return histogram[histogram.size() - 1];
}
cv::Mat ImgHelper::convertUcharRGBToFloatGrayImage(const cv::Mat& rgbImage)
{
    cv::Mat grayValImage(rgbImage.size(), CV_32FC1);

#pragma omp parallel for
    for (int yIdx = 0; yIdx < rgbImage.rows; yIdx++)
    {
        const cv::Vec3b* dataBGR = rgbImage.ptr<cv::Vec3b>(yIdx);
        float* dataGray = grayValImage.ptr<float>(yIdx);

        for (int xIdx = 0; xIdx < rgbImage.cols; xIdx++)
        {
            dataGray[xIdx] = 0.00392157f * (dataBGR[xIdx][0] * 0.11f + dataBGR[xIdx][1] * 0.59f + dataBGR[xIdx][1] * 0.3f); 
        }
    }

    return grayValImage;
}

void ImgHelper::weightedMedianFilterSequentiell(const int& minDisparity, const int& maxDisparity, const int& k, const cv::Mat& grayValInputImage, cv::Mat& inOutImage)
{
    int radius = k * 0.5;
    float sigmaSpace = 9.f * 9.f * 0.25;
    float sigmaGrayVal = 25.5f * 25.5f;
    float invertedSigmaSpace = 1.f / sigmaSpace;
    float invertedSigmaGrayVal = 1.f / sigmaGrayVal;

    std::vector<std::vector<float>> distSpaceSquared(k, std::vector<float>(k));
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < k; j++)
        {
            distSpaceSquared[i][j] = i * i + j * j;
            distSpaceSquared[i][j] *= invertedSigmaSpace;
        }
    }

    std::vector<const uchar*> filterDepthRows(k);
    std::vector<const float*> filterGrayValRows(k);

    //first set Borderpixel if non existing#
    std::vector<float> validNeighbours(maxDisparity - minDisparity, 0.f);
    for (int yIdx = 0; yIdx < inOutImage.rows-k +1 ; yIdx++)
    {
        //fill filterDepthRows and filterGrayValRows:
        for (int i = 0; i < filterDepthRows.size(); i++)
        {
            int imageIdx = yIdx + i;
            filterDepthRows[i] = inOutImage.ptr<uchar>(imageIdx);
            filterGrayValRows[i] = grayValInputImage.ptr<float>(imageIdx);
        }

        uchar* inOutData = inOutImage.ptr<uchar>(yIdx+radius);
        const float* grayValData = grayValInputImage.ptr<float>(yIdx);

        for (int xIdx = 0; xIdx < inOutImage.cols - k + 1; xIdx++)
        {
            if (!inOutData[xIdx + radius]) //<- check if occlusion pixel
            {
                //search valid neighbours:
                for (int i = 0; i < filterDepthRows.size(); i++)
                {
                    for (int j = 0; j < k; j++)
                    {
                        int imageXIdx = xIdx + j ;
                        uchar neighbourVal = filterDepthRows[i][imageXIdx];
                        if (neighbourVal)
                        {
                            float grayValDist = filterGrayValRows[i][imageXIdx] - grayValData[xIdx];
                            float weightVal = distSpaceSquared[i][j] + 255.0 * grayValDist * 255.0 * grayValDist * invertedSigmaGrayVal;
                            validNeighbours[neighbourVal - minDisparity] += std::exp(-weightVal);
                        }
                    }
                }

                float sum = cumSum(validNeighbours);

                if (sum)
                {
                    float thresVal = sum * 0.5f;
                    int medianIndex = std::distance(validNeighbours.begin(), std::find_if(validNeighbours.begin(), validNeighbours.end(), [&thresVal](float& element) { return (thresVal <= element); }));
                    inOutData[xIdx+ radius] = medianIndex + minDisparity;
                    std::fill(validNeighbours.begin(), validNeighbours.end(), 0.f);
                }
            }
        }
    }
}
