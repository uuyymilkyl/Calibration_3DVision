#ifndef _MANGLE2MATRIXTOOL_H
#define _MANGLE2MATRIXTOOL_H
#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include <iostream>
#include <string>
#include <fstream>


//工具函数 用于反复调用
inline static void EulZXZToQuat(cv::Mat& _eulAngle, cv::Mat& _quatResult)
{


    double A = cv::fastAtan2(std::sin(A), std::cos(A)) * CV_PI / 180.0;
    double B = cv::fastAtan2(std::sin(B), std::cos(B)) * CV_PI / 180.0;
    double C = cv::fastAtan2(std::sin(C), std::cos(C)) * CV_PI / 180.0;

    // 计算欧拉角的一半
    A /= 2.0;
    B /= 2.0;
    C /= 2.0;

    // 计算三个轴的sin和cos值
    double sa = std::sin(A);
    double ca = std::cos(A);
    double sb = std::sin(B);
    double cb = std::cos(B);
    double sx = std::sin(C);
    double cx = std::cos(C);

    // 计算四元数的四个分量
    double qw = ca * cb * cx + sa * sb * sx;
    double qx = sa * cb * cx - ca * sb * sx;
    double qy = ca * sb * cx + sa * cb * sx;
    double qz = ca * cb * sx - sa * sb * cx;

    // 构造四元数矩阵
    cv::Mat quaternion = (cv::Mat_<double>(4, 1) << qw, qx, qy, qz);


}

static inline std::vector<double> ConvMatTovFloat(const std::vector<cv::Mat>& matVector) {
    std::vector<double> floatVector;

    for (const auto& mat : matVector) {
        if (mat.isContinuous()) {
            floatVector.insert(floatVector.end(), mat.ptr<double>(), mat.ptr<double>() + mat.total());
        }
        else {
            for (int i = 0; i < mat.rows; ++i) {
                const double* rowPtr = mat.ptr<double>(i);
                floatVector.insert(floatVector.end(), rowPtr, rowPtr + mat.cols);
            }
        }
    }

    return floatVector;
}
#endif // !_MANGLE2MATRIXTOOL_H
