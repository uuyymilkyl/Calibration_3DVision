#ifndef _MHAHDEYECALIBRATION_H_
#define _MHANDEYECALIBRATION_H_

#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include <iostream>
#include <string>

#include"CalibrationDefine.h"
#include "MAngle2MatrixTool.hpp"
//
//手眼标定的类 眼在手上
class MCaliHandEye
{
public:
	MCaliHandEye();
	~MCaliHandEye();

	void GetChessBoardMatirx(std::string& _CaliImg, std::vector<cv::Mat> &_R_ObCamPoseMat, std::vector<cv::Mat>  &_T_ObCamPoseMat);

	std::vector<cv::Mat>GetArmPoseMatrix(std::string& _txtFile,std::vector<cv::Mat>& _R_Pose, std::vector<cv::Mat>& _T_Pose);

	std::vector<cv::Mat>CalTransMat();

	void EyeHandCalibration(std::vector<double> _RobotRVec, 
		                    std::vector<double> _RobotTVec,
		                    std::vector<double> _CamRVec, 
		                    std::vector<double> _CamTVec,
		                    cv::Mat& _Cam2EndR, 
		                    cv::Mat& _Cam2EndT);

	void XYZ2TransMat(double _DeltaX, double _DeltaY, double _DeltaZ, cv::Mat _TransMat);

	void Euler2RotateMat(double _DegreeA, double _DegreeB, double _DegreeC, cv::Mat& _RotateMat, int _iType);

	cv::Mat RotateXMat(double _Degree);

	cv::Mat RotateYMat(double _Degree);

	cv::Mat RotateZMat(double _Degree);
private:

	std::vector<cv::Mat> m_R_vecMatrix, m_T_vecMatrix;
	std::vector<cv::Mat> m_R_ArmPoseMatrix, m_T_ArmPoseMatrix;
};

#endif // !_MHAHDEYECALIBRATION_H_
