#ifndef  _LASER_CALIBRATION_H
#define  _LASER_CALIBRATION_H

#include<opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include "MCamCalibration.hpp"

class MCalibLaser
{
public:
	MCalibLaser();
	~MCalibLaser();
	// 加载光平面标定图像
	void LoadLaserImage(std::string &_LaserImgDir ,std::vector<cv::Mat>& _CaliCamImg, std::vector <cv::Mat>& _CaliLasImg, cv::Mat& _intrinsicMatrix, cv::Mat& _coeffsMatrix);
	
	// 标定，获取无激光图的对应外参
	void CalibraCamForLaserImg(std::vector<cv::Mat>& _CaliCamImg, std::vector< cv::Mat> &_Rves, std::vector<cv::Mat> &_Tves);
	
	//Steger算法获取激光轮廓
	static std::vector<cv::Point2f> GetLaserPoints_Steger(cv::Mat& src, int gray_Thed, int Min, int Max, int Type);

	//Sobel算法获取激光轮廓
	static std::vector<cv::Point2f> GetLaserPoints_Sobel(const cv::Mat& inputImage);

	// 获取通过标定板获取激光点集的范围
	void GetCaliRangeAndFitLine(cv::Mat& _inputImg, int &xmin, int &ymin, int &xmax, int &ymax);

	//通过标定板角点范围筛选激光轮廓点
	std::vector<cv::Point2f> FilterLaserPointsByRange(std::vector<cv::Point2f> _LaserPoints,int xmin,int ymin,int xmax,int ymax);

	//按行拟合标定板的角点
	std::vector<cv::Vec4f> FitChessBoardLines(const std::vector<cv::Point2f>& corners, cv::Mat& image,std::vector<std::vector<cv::Point2f>> &_Line_Points);

	// （每个点）图像坐标转相机坐标
	void PointToCameraPoint(cv::Point2f _vptImg, cv::Point3f& _vptCam, const cv::Mat _matRvecs, const cv::Mat _matTvecs, const cv::Mat _matIntrinsics);

	cv::Point2f getCrossPoint(const cv::Vec4f& line1, const cv::Vec4f& line2);

	//拟合光平面
	void FitLaserPlane(std::vector<cv::Point3f> _vecPoints3ds, cv::Mat& _mat_plane);

	//基于光平面系数 将图像坐标系转为相机坐标系（2d -> 3d)
	static void ImgPtsToCamFrame(const cv::Mat _LaserPlane_Coeffs, const cv::Mat _Intrinsic_Matrix, cv::Mat _Dist_Coeff,
		std::vector<cv::Point2f> _vec_ptSrc, std::vector<cv::Point3f>& _vec_ptDst);
	//畸变矫正
	static void DistortPoints(std::vector<cv::Point2f> _ptSrc, std::vector<cv::Point2f>& _ptDst, const cv::Mat _matIntrinsics, const cv::Mat _matDistCoeff);

	static void Writedown(std::vector<cv::Point3f>& _vec_ptDst,std::string _name);


	
private:

	cv::Mat BaseLayExtrinxMatrix;
	std::vector<std::vector<cv::Point2f>> m_imagePoints;
	
};





#endif // ! _LASER_CALIBRATION_H
