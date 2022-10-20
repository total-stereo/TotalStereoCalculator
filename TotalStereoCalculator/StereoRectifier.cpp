#include "StereoRectifier.h"

#include <opencv2/calib3d.hpp>


cv::Rect determineOverlappingROI(const cv::Rect& roi1, const cv::Rect& roi2)
{
    int yTop = std::max(roi1.y, roi2.y);
    int yBottom = std::min(roi1.br().y, roi2.br().y);
    int xTop = std::max(roi1.x, roi2.x);
    int xBottom = std::min(roi1.br().x, roi2.br().x);
    int heightCroppedImage = yBottom - yTop;
    int widthCroppedImage = xBottom - xTop;

    cv::Rect overlappingROI(xTop, yTop, widthCroppedImage, heightCroppedImage);

    return overlappingROI;
}

cv::Rect getTransformedImageDimensions(const std::vector<cv::Point3f>& imCoor, const int& maxLengthX, const int& maxLengthY)
{
    if (imCoor.size() != 4)
    {
        return cv::Rect();
    }

    //calculate  new x-Dimensions:
    int xTopLeftPos = std::max(0, static_cast<int>(imCoor[0].x / imCoor[0].z + 0.5f));
    int xBottomLeftPos = std::max(0, static_cast<int>(imCoor[3].x / imCoor[3].z + 0.5f));

    int xTopRightPos = std::min(maxLengthX, static_cast<int>(imCoor[1].x / imCoor[1].z - 0.5f));
    int xBottomRightPos = std::min(maxLengthX, static_cast<int>(imCoor[2].x / imCoor[2].z - 0.5f));

    int maxXLeftSide = std::max(xTopLeftPos, xBottomLeftPos);
    int minXRightSide = std::min(xTopRightPos, xBottomRightPos);

    int width = minXRightSide - maxXLeftSide;

    //calculate  new y-Dimensions:
    int yTopLeftPos = std::max(0, static_cast<int>(imCoor[0].y / imCoor[0].z + 0.5f));
    int yBottomLeftPos = std::max(0, static_cast<int>(imCoor[3].y / imCoor[3].z + 0.5f));

    int yTopRightPos = std::min(maxLengthY, static_cast<int>(imCoor[1].y / imCoor[1].z - 0.5f));
    int yBottomRightPos = std::min(maxLengthY, static_cast<int>(imCoor[2].y / imCoor[2].z - 0.5f));

    int maxYTop = std::max(yTopLeftPos, yTopRightPos);
    int minYBottom = std::min(yBottomLeftPos, yBottomRightPos);

    int height = minYBottom - maxYTop;

    return cv::Rect(maxXLeftSide, maxYTop, width, height);
}


std::tuple<cv::Mat, cv::Mat, int, int> StereoRectifier::compute(const cv::Mat& leftImage, const cv::Mat rightImage)
{
    //1. detect features in left and right image
    _siftOperator->detectAndCompute(leftImage, cv::Mat(), _keyPointsLeftImage, _descriptorLeftImage);
    _siftOperator->detectAndCompute(rightImage, cv::Mat(), _keyPointsRightImage, _descriptorRightImage);

    // 2. match features 
    std::vector< std::vector<cv::DMatch> > knnMatches;
    _matchingOperator->knnMatch(_descriptorLeftImage, _descriptorRightImage, knnMatches, 2);

    const float ratioThresh = 0.75f;
    std::vector<cv::DMatch> goodMatches;
    for (size_t i = 0; i < knnMatches.size(); i++)
    {
        if (knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance)
        {
            goodMatches.push_back(knnMatches[i][0]);
        }
    }

    std::vector<cv::Point2f> pointsLeftImage(goodMatches.size());
    std::vector<cv::Point2f> pointsRightImage(goodMatches.size());
    for (size_t i = 0; i < goodMatches.size(); i++)
    {
        // Get the keypoints from the good matches
        pointsLeftImage[i] = _keyPointsLeftImage[goodMatches[i].queryIdx].pt;
        pointsRightImage[i] = _keyPointsRightImage[goodMatches[i].trainIdx].pt;
    }

    // 3. Determine Fundamental matrix
    std::vector<uchar> outPutMask;
    cv::Mat fundamentalMatrix = cv::findFundamentalMat(pointsRightImage, pointsLeftImage, outPutMask, cv::RANSAC, 0.75);

    std::vector<cv::Point2f> pointsLeftFiltered, pointsRightFiltered;
    for (int i = 0; i < outPutMask.size(); i++)
    {
        if (outPutMask[i])
        {
            pointsLeftFiltered.push_back(pointsLeftImage[i]);
            pointsRightFiltered.push_back(pointsRightImage[i]);
        }
    }

    // 4. Determine transformation mats 
    cv::Mat rightH, leftH;
    cv::stereoRectifyUncalibrated(pointsRightFiltered, pointsLeftFiltered, fundamentalMatrix, leftImage.size(), rightH, leftH, 0.75);

    // 5. Transform/Rectify images
    cv::Mat rectifiedLeftImage, rectifiedRightImage;
    cv::warpPerspective(leftImage, rectifiedLeftImage, leftH, leftImage.size());
    cv::warpPerspective(rightImage, rectifiedRightImage, rightH, rightImage.size());

    // 6. Crop images so that there is no empty border

    //cornerPositions in "old" image (in homogeneous coordinates):
    int maxXCoor = leftImage.cols - 1;
    int maxYCoor = leftImage.rows - 1;
    std::vector<cv::Point3f> vec = { cv::Point3f(0,0,1), cv::Point3f(maxXCoor,0,1),cv::Point3f(maxXCoor,maxYCoor,1),cv::Point3f(0,maxYCoor,1) };

    //cornerPositions in rectified image (in homogeneous coordinates):
    std::vector<cv::Point3f> imCoorLeft(4), imCoorRight(4);
    cv::transform(vec, imCoorLeft, leftH);
    cv::transform(vec, imCoorRight, rightH);

    //ROI for each image, so that there is no empty border visible in the image
    cv::Rect leftImageCornerPos = getTransformedImageDimensions(imCoorLeft, maxXCoor, maxYCoor);
    cv::Rect rightImageCornerPos = getTransformedImageDimensions(imCoorRight, maxXCoor, maxYCoor);

    //Determine overlapping rectangles
    cv::Rect overlappingROI =  determineOverlappingROI(leftImageCornerPos, rightImageCornerPos);

    cv::Mat rectifiedLeftImageCropped = rectifiedLeftImage(overlappingROI);
    cv::Mat rectifiedRightImageCropped = rectifiedRightImage(overlappingROI);

    //7. calculate min and max disparity 
    std::vector<cv::Point3f> pointsLeftFilteredHomogenous(pointsLeftFiltered.size()), pointsRightFilteredHomogenous(pointsLeftFiltered.size());
    for (int i = 0; i < pointsLeftFiltered.size(); i++)
    {
        pointsLeftFilteredHomogenous[i] = cv::Point3f(pointsLeftFiltered[i].x, pointsLeftFiltered[i].y, 1.f);
        pointsRightFilteredHomogenous[i] = cv::Point3f(pointsRightFiltered[i].x, pointsRightFiltered[i].y, 1.f);
    }

    std::vector<cv::Point3f> transformedFeaturesLeft(pointsLeftFiltered.size()), transformedFeaturesRight(pointsLeftFiltered.size());
    cv::transform(pointsLeftFilteredHomogenous, transformedFeaturesLeft, leftH);
    cv::transform(pointsRightFilteredHomogenous, transformedFeaturesRight, rightH);

    std::vector<float> xDistances(pointsLeftFiltered.size());

    for (int i = 0; i < pointsLeftFiltered.size(); i++)
    {
        xDistances[i] = transformedFeaturesLeft[i].x / transformedFeaturesLeft[i].z - transformedFeaturesRight[i].x / transformedFeaturesRight[i].z; // -links - rechts
    }
 //   int xIdx2 = xIdx - d; //xIdx2 == rechts -> linkes Bild vorne


    auto [minDisparity, maxDisparity] = std::minmax_element(xDistances.begin(), xDistances.end());

    double factor = 1.0; // 1.1;
    int minD = factor*static_cast<int>(factor*(*minDisparity - 0.5));
    int maxD = static_cast<int>(factor * (*maxDisparity + 0.5));

   /* int borderRange = abs(static_cast<int>(0.5 * minD));

    cv::Rect shiftROI1 = cv::Rect(borderRange , 0, rectifiedLeftImageCropped.cols - borderRange, rectifiedLeftImageCropped.rows);
    cv::Rect shiftROI2 = cv::Rect(0, 0, rectifiedLeftImageCropped.cols - borderRange, rectifiedLeftImageCropped.rows);

    cv::Rect leftAlignROI = minD < 0 ? shiftROI2 : shiftROI1;
    cv::Rect rightAlignROI = minD < 0 ? shiftROI1 : shiftROI2;
    */
    cv::Mat leftAlignedImage = rectifiedLeftImageCropped; // (leftAlignROI);
    cv::Mat rightAlignedImage = rectifiedRightImageCropped; // (rightAlignROI);

    return { leftAlignedImage , rightAlignedImage , minD, maxD };
}
