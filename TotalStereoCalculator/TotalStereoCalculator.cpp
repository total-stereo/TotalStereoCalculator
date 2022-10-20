// TotalStereoCalculator.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "StereoRectifier.h"
#include "DepthMapCalculator.h"

//https://www.3dtv.at/Knowhow/AnaglyphComparison_en.aspx
cv::Mat computeAnaglyphImage(const cv::Mat& leftImage, const cv::Mat& rightImage)
{
    cv::Mat anaglyphImage = cv::Mat(leftImage.size(), leftImage.type());

#pragma omp parallel for
    for (int yIdx = 0; yIdx < leftImage.rows; yIdx++)
    {
        const cv::Vec3b* dataLeft = leftImage.ptr<cv::Vec3b>(yIdx);
        const cv::Vec3b* dataRight = rightImage.ptr<cv::Vec3b>(yIdx);
        cv::Vec3b* dataAnaglyph = anaglyphImage.ptr<cv::Vec3b>(yIdx);

        for (int xIdx = 0; xIdx < leftImage.cols; xIdx++)
        {
            dataAnaglyph[xIdx] = cv::Vec3b(dataLeft[xIdx][0], dataLeft[xIdx][1], dataRight[xIdx][2]);

        }
    }

    return anaglyphImage;
}


int main()
{
	std::string imageFolderInput = "E:\\Hackathon\\Stereodias-Schreiber-Serie3\\Processed\\"; 
	std::string imageFolderOutput = "E:\\Hackathon\\Stereodias\\";

    std::vector<std::string> fileNamesLeftImage;
    std::vector<std::string> fileNamesRightImage;
    std::vector<std::string> fileNamesBase;

    for (const auto& entry : std::filesystem::directory_iterator(imageFolderInput))
    {
        auto filePath = entry.path();
        if (filePath.extension().string().compare(".png") == 0)
        {
            if (filePath.stem().string().find("left") != std::string::npos)
            {
                fileNamesLeftImage.push_back(filePath.filename().string());
                std::string basefileName = filePath.stem().string();
                basefileName.resize(basefileName.size() - 4);
                fileNamesBase.push_back(basefileName);
            }
            else
            {
                fileNamesRightImage.push_back(filePath.filename().string());
            }
        }

    }

    for (int i = 0; i < fileNamesBase.size(); i++) 
    {
        //read left and rightImage
        cv::Mat imageRight = cv::imread(imageFolderInput + fileNamesRightImage[i]);
        cv::Mat imageLeft = cv::imread(imageFolderInput + fileNamesLeftImage[i]);

        //downsample image
        cv::Mat resizedLeftImage;
        cv::resize(imageLeft, resizedLeftImage, cv::Size(), 0.5, 0.5, cv::INTER_AREA);

        cv::Mat resizedRightImage;
        cv::resize(imageRight, resizedRightImage, cv::Size(), 0.5, 0.5, cv::INTER_AREA);

        
        StereoRectifier stereoRectifier;
        auto [rectifiedLeftImage, rectifiedRightImage,minDisparity, maxDisparity] = stereoRectifier.compute(resizedLeftImage, resizedRightImage);

        DepthMapCalculator depthMapCalculator;
        cv::Mat depthImage = depthMapCalculator.compute(rectifiedLeftImage, rectifiedRightImage, minDisparity, maxDisparity);
       
        cv::Mat anaglyph = computeAnaglyphImage(rectifiedLeftImage, rectifiedRightImage);
        cv::imwrite(imageFolderOutput + "left\\" + fileNamesRightImage[i], rectifiedLeftImage);
        cv::imwrite(imageFolderOutput + "right\\" + fileNamesLeftImage[i], rectifiedRightImage);
        cv::imwrite(imageFolderOutput + "depth\\"+ fileNamesBase[i]+"depth.png", depthImage);

    }


}
