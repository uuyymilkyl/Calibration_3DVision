#ifndef _MCALIBRATION_H_
#define _MCAlIBRATION_H_


#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include <iostream>
#include <string>

class MCalibCam
{
public:
	MCalibCam();
	~MCalibCam();

    // 设置标定棋盘格的尺寸
	void SetBoardSize(int WidthLatticeNum, int HeightLatticeNum, float SquarSize); 

    // 加载图像及棋盘格数据
	void LoadChessBoardData(std::string& _ImgDir); 

    // 相机标定主流程 - 得到相机内参 - 畸变矩阵 - 每符图像的平移矩阵 - 旋转矩阵
    void CalibrationOfCamera( cv::Mat &_intrinxMatrix, cv::Mat &_coeffsMatrix);

    // 计算世界坐标系的坐标 
    void CalEConerCoordinationInCamera(std::vector<cv::Point2f>_corner,int imageindex, cv::Mat Rvecs, cv::Mat Tvecs);  //计算每个角点到相机的Z轴距离

    cv::Point3f CalPlaneLineIntersectPoint(const cv::Vec3d& planeNormal, const cv::Point3f& planePoint,
        const cv::Vec3f& lineDirection, const cv::Point3f& linePoint);



private:
    int m_WidthLattinceNum; //标定板横向角点个数（格数-1）
    int m_HeightLatticeNum; //标定板纵向角点个数（格数-1）
    float m_SquarSize;      //标定板格边长（mm)
    int m_PixelSize;        //像元尺寸（相机sensor决定）（mm)

    cv::Size m_ImgSize;     //相机的图像分辨率
    std::vector<cv::String> m_ImgList;

    cv::Mat m_IntrinxMatrix;  //相机（内）参数
    cv::Mat m_CoeffsMatrix;   //畸变参数

    std::vector<std::vector<cv::Point2f>> m_imagePoints;
    std::vector<std::vector<cv::Point3f>> m_objectPoints;
};


static bool haveSameXCoordinate(const cv::Point2f& p1, const cv::Point2f& p2) {
    return p1.x == p2.x;
}

// Function to compare points based on y coordinate
static bool comparePointsByY(const cv::Point2f& p1, const cv::Point2f& p2) {
    return p1.y < p2.y;
}

// Function to find median of y coordinates of adjacent points with the same x coordinate
static std::vector<cv::Point2f> findMedianPoints(const std::vector<cv::Point2f>& points) {
    std::vector<cv::Point2f> resultPoints;

    for (size_t i = 0; i < points.size(); ++i) {
        if (i == 0 || !haveSameXCoordinate(points[i], points[i - 1])) {
            // If it's the first point or if current point has different x coordinate than the previous point
            resultPoints.push_back(points[i]);
        }
        else {
            // Find adjacent points with same x coordinate
            size_t startIndex = i;
            while (i < points.size() && haveSameXCoordinate(points[i], points[startIndex])) {
                ++i;
            }

            // Collect y coordinates of adjacent points
            std::vector<float> yCoordinates;
            for (size_t j = startIndex; j < i; ++j) {
                yCoordinates.push_back(points[j].y);
            }

            // Calculate median of y coordinates
            float medianY = 0.0f;
            if (!yCoordinates.empty()) {
                std::sort(yCoordinates.begin(), yCoordinates.end());
                size_t mid = yCoordinates.size() / 2;
                if (yCoordinates.size() % 2 == 0) {
                    medianY = (yCoordinates[mid - 1] + yCoordinates[mid]) / 2.0f;
                }
                else {
                    medianY = yCoordinates[mid];
                }
            }

            // Create a new point with same x coordinate and median y coordinate
            resultPoints.emplace_back(points[startIndex].x, medianY);
            --i; // Adjust the index as the loop will increment it again
        }
    }

    return resultPoints;
}


inline static std::vector<cv::Point2f> mergePointsWithEqualY(std::vector<cv::Point2f>& ContoursPoints) {
    // 创建一个 map 以 x 坐标作为键，将点按照相同的 x 坐标分组
    std::map<float, std::vector<cv::Point2f>> pointsByX;

    // 将点按照 y 坐标分组
    for (const auto& point : ContoursPoints) {
        pointsByX[point.x].push_back(point);
    }

    std::vector<cv::Point2f> mergedPoints;
    std::vector<cv::Point2f> EveYPoints;
    // 对每个相同 y 坐标的点集合进行处理
    for (const auto& pair : pointsByX) 
    {
        const auto& points = pair.second;


        EveYPoints = findMedianPoints(points);
        mergedPoints.insert(mergedPoints.end(), EveYPoints.begin(), EveYPoints.end());

    }
    return mergedPoints;
}

inline static void ExtractRedLaserContour(const cv::Mat& inputImage, std::vector<cv::Point2f>& outputContourPoints) {
    // 转换输入图像到HSV颜色空间
    cv::Mat hsvImage;
    cv::cvtColor(inputImage, hsvImage, cv::COLOR_BGR2HSV);

    // 将红色激光线部分提取出来
    cv::Mat redMask;
    cv::Mat whiteMask;
    //cv::inRange(inputImage, cv::Scalar(0, 0, 25), cv::Scalar(70, 70, 255), redMask);
    cv::inRange(inputImage, cv::Scalar(0, 0, 100), cv::Scalar(70, 70, 255), redMask);
    cv::inRange(inputImage, cv::Scalar(140, 110, 220), cv::Scalar(255, 255, 255), whiteMask);
    cv::Mat redContourImage = cv::Mat::zeros(inputImage.size(), CV_8UC1);



    // 根据dx、dy和阈值选择最佳唯一的轮廓点
    /*
    for (size_t i = 0; i < contours.size(); ++i) {
        std::vector<cv::Point>& contour = contours[i];
        if (contour.size() < 2) {
            continue;
        }
        for (size_t j = 0; j < contour.size(); ++j) {
            int x = contour[j].x;
            int y = contour[j].y;

            if (x % dx == 0 && y % dy == 0) {
                cv::Vec3b pixel = hsvImage.at<cv::Vec3b>(y, x);
                if (pixel[0] > redThreshold) {
                    outputContourPoints.push_back(cv::Point2f(x, y));
                }
            }
        }
    }*/

    std::vector<cv::Point2f> ContoursPoints;
    for (int y = 0; y < redMask.rows; ++y)
    {
        for (int x = 0; x < redMask.cols; ++x)
        {
            if (redMask.at<uchar>(y, x) == 255)
            {
                ContoursPoints.push_back(cv::Point2f(x, y));
            }
            if (whiteMask.at<uchar>(y, x) == 255)
            {
                ContoursPoints.push_back(cv::Point2f(x, y));
            }
        }
    }
    

    std::vector<cv::Point2f> ContoursPoints_Merge;

    ContoursPoints_Merge= mergePointsWithEqualY(ContoursPoints);

    outputContourPoints = ContoursPoints_Merge;
    for (int i = 0; i < ContoursPoints_Merge.size(); i++)
    {
        cv::circle(inputImage, ContoursPoints_Merge[i], 1, cv::Scalar(255, 0, 0), 2, 6, 0);
    }
}

#endif