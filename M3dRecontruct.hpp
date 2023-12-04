#ifndef _M3DRecontruct_H_
#define _M3DRecontruct_H_

#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include "MLasCalibration.hpp"
#include "MHandEyeCalibration.hpp"

#define CamCaliXmlPath
#define LaserCaliXmlPath
#define HandEyeCaliXmlPath

class M3DRecontruct
{
public:
	M3DRecontruct();
	~M3DRecontruct();
	//画出点云 
	static void WritedownCloudPoints(std::vector<cv::Point3f>& _vec_ptDst, std::string _name);

	//加载标定信息
	void LoadCalibraionParam(std::string& _CamCaliXmlDir, std::string& _LaserCaliXmlDir, std::string& _HandEyeCaliXmlDir);

	//通过视频计算点云
	void VideoRecontruct(std::string& _videopath);

	//通过拍摄照片（测试）连接点云并得到变换姿态
	void TestRecontructByFrame(std::string &_ImagePath,std::string &_ArmPosePath);

	//计算每个点云的世界坐标系
	std::vector<cv::Point3f> AccumulateFrameRecontruct(cv::Mat &_frame, cv::Mat& _HandEyeMatrix_R , cv::Mat& _HandEyeMatrix_T, cv::Mat& _CurrenArmPose);


	// 子功能 
	cv::Mat eulerToRotationMatrix(const cv::Mat& angles);

	cv::Mat buildPoseMatrix(const cv::Mat& t, const cv::Mat& angles);


	//测试用：到时候替换成相机就不用这个接口
	std::vector<cv::Mat> LoadArmPoseByTxt(std::string& _Dir);
private:
	cv::Mat m_IntrinxMatrix;     ///< 相机内参
	cv::Mat m_DistortCoeffs;     ///< 畸变系数
	cv::Mat m_LaserPlaneCoeffs;  ///< 光平面系数
	cv::Mat m_HandEyeMatrix_R;   ///< 手眼标定的R
	cv::Mat m_HandEyeMatrix_T;   ///< 手眼标定的T
	cv::Mat m_LastArmPose;       ///< 上一帧的机械臂姿态

	
	int m_frame_count;
};


#endif // !_M3D_Recontruct_H_
