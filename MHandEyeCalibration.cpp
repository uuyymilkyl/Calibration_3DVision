
#include "MHandEyeCalibration.hpp"
#include <iostream>
#include <string>
#include <fstream>

MCaliHandEye::MCaliHandEye()
{
	std::string ArmPoseDir = "./CaliHandEye/1128.txt";
	std::string CaliPicDir = "./CaliHandEye/HandEye1128/*.jpg";

	std::vector<cv::Mat> R_ArmPoseMat,T_ArmPoseMat;
	GetArmPoseMatrix(ArmPoseDir ,R_ArmPoseMat, T_ArmPoseMat); 


	std::vector<cv::Mat> R_ObCamPoseMat, T_ObCamPoseMat;
	GetChessBoardMatirx(CaliPicDir, R_ObCamPoseMat, T_ObCamPoseMat);


	
	std::vector<double> R_ArmPose, T_ArmPose, R_ObCamPose, T_ObCamPose;
	R_ArmPose = ConvMatTovFloat(R_ArmPoseMat);
	T_ArmPose = ConvMatTovFloat(T_ArmPoseMat);
	R_ObCamPose = ConvMatTovFloat(R_ObCamPoseMat);
	T_ObCamPose = ConvMatTovFloat(T_ObCamPoseMat);

	cv::Mat R_Arm2CamMat;
	cv::Mat T_Arm2CamMat;

	EyeHandCalibration(R_ArmPose,T_ArmPose,R_ObCamPose,T_ObCamPose,R_Arm2CamMat,T_Arm2CamMat);


}

MCaliHandEye::~MCaliHandEye()
{
}

void MCaliHandEye::GetChessBoardMatirx(std::string& _CaliImgDir, std::vector<cv::Mat> &_R_ObCamPoseMat, std::vector<cv::Mat>  &_T_ObCamPoseMat)
{

	std::vector<cv::String> ImgList;
	cv::glob(_CaliImgDir, ImgList);

	cv::Size ImgSize;

	std::vector<std::vector<cv::Point3f>> objectPoints;
	std::vector<cv::Point3f> realChessboardData;


	float chesssize = 5.0f;
	for (int i = 0; i < 8; ++i)
	{
		for (int j = 0; j < 11; ++j)
		{
			realChessboardData.push_back(cv::Point3f(j * chesssize, i * chesssize, 0.0f));
		}
	}

	for (size_t i = 0; i < ImgList.size(); ++i)
	{
			objectPoints.push_back(realChessboardData);
	}


	std::vector<cv::Point2f> nCorners;
	std::vector<std::vector<cv::Point2f>> imagePoints;
	for (int i = 0; i < ImgList.size(); ++i)
	{
			cv::Mat image = cv::imread(ImgList[i]);
			cvtColor(image, image, cv::COLOR_BGR2GRAY);

			//找一张图像上的棋盘格角点
			bool found = findChessboardCorners(image, cv::Size(11, 8), nCorners);

			if (found)
			{
				// 亚像素定位精确找寻角点 
				cornerSubPix(image, nCorners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
				// 将本图中找到的角点集放回vector中
				imagePoints.push_back(nCorners);
				// 画出角点（可视化操作）
				drawChessboardCorners(image, cv::Size(11, 8), nCorners, found);
				ImgSize = image.size();
			}
	}

		cv::Mat intrinxMatrix, coeffsMatrix;
		std::vector<cv::Mat> Rvecs, Tvecs;
		calibrateCamera(objectPoints, imagePoints, ImgSize, intrinxMatrix, coeffsMatrix, Rvecs, Tvecs);

		//对每张标定板求出他的对应外参
		cv::Mat rotationMatrix;//3*3

		std::vector<cv::Mat> n_Rvecs, n_Tvecs;
		for (int i = 0; i < Rvecs.size(); i++)
		{
			
			cv::Rodrigues(Rvecs[i], rotationMatrix);

			double r00 = rotationMatrix.at<double>(0, 0);
			double r01 = rotationMatrix.at<double>(0, 1);
			double r02 = rotationMatrix.at<double>(0, 2);
			double r10 = rotationMatrix.at<double>(1, 0);
			double r11 = rotationMatrix.at<double>(1, 1);
			double r12 = rotationMatrix.at<double>(1, 2);
			double r20 = rotationMatrix.at<double>(2, 0);
			double r21 = rotationMatrix.at<double>(2, 1);
			double r22 = rotationMatrix.at<double>(2, 2);

			double r03 = Tvecs[i].at<double>(0, 0);
			double r13 = Tvecs[i].at<double>(1, 0);
			double r23 = Tvecs[i].at<double>(2, 0);

			double r30 = 0;
			double r31 = 0;
			double r32 = 0;
			double r33 = 1;

			cv::Mat R_PoseMatrix = (cv::Mat_<double>(3, 3) << r00,r01,r02,r10,r11,r12,r20,r21,r22);// 3x3
			cv::Mat T_PoseMatrix = (cv::Mat_<double>(1, 3) << r03, r13, r23); //1x3

			n_Rvecs.push_back(R_PoseMatrix);
			n_Tvecs.push_back(T_PoseMatrix);
		}
		
		_R_ObCamPoseMat = Rvecs;
		_T_ObCamPoseMat = Tvecs;

	
}

std::vector<cv::Mat> MCaliHandEye::GetArmPoseMatrix(std::string& _txtFile,std::vector<cv::Mat> &_R_Pose, std::vector<cv::Mat>& _T_Pose)
{
	std::vector<cv::Mat> ArmPoseMatrix;
	std::vector<cv::Mat> R_ArmPose;
	std::vector<cv::Mat> T_ArmPose;
	std::ifstream file(_txtFile);

	if (!file.is_open())
	{
		std::cerr << "Error opening file: " << _txtFile << std::endl;
		return ArmPoseMatrix;
	}

	std::string line;
	while (std::getline(file, line))
	{
		std::istringstream iss(line);
		cv::Mat rowMat(1, 6, CV_64F);
		cv::Mat rowRMat(1, 3, CV_64F);
		cv::Mat rowTMat(1, 3, CV_64F);

		for (int i = 0; i < 6; ++i) 
		{
			double value;
			if (!(iss >> value))
			{
				std::cerr << "Error reading value from line: " << line << std::endl;
				break;
			}
			rowMat.at<double>(0, i) = value;
			if (i < 3) 
			{
				rowTMat.at<double>(0, i) = value;
			}
			else 
			{
				rowRMat.at<double>(0, i-3) = value;
			}

		}

		R_ArmPose.push_back(rowRMat);
		T_ArmPose.push_back(rowTMat);
		ArmPoseMatrix.push_back(rowMat);

	}

	_R_Pose = R_ArmPose;
	_T_Pose = T_ArmPose;
	file.close();


	return ArmPoseMatrix;
}

std::vector<cv::Mat> MCaliHandEye::CalTransMat()
{
	return std::vector<cv::Mat>();
}

void MCaliHandEye::EyeHandCalibration(std::vector<double> _RobotRVec, std::vector<double> _RobotTVec, std::vector<double> _CamRVec, std::vector<double> _CamTVec, cv::Mat& _Cam2EndR, cv::Mat& _Cam2EndT)
{
	std::vector<cv::Mat> RobotRMat;
	std::vector<cv::Mat> RobotTMat;
	std::vector<cv::Mat> CamRMat;
	std::vector<cv::Mat> CamTMat;

	for (int i = 0; i < _RobotRVec.size(); i += 3)
	{
		cv::Mat mat = cv::Mat(3, 3, CV_64FC1);
		cv::Mat transmat = (cv::Mat_<double>(3, 1) << _RobotTVec[i], _RobotTVec[i + 1], _RobotTVec[i + 2]);
		Euler2RotateMat(_RobotRVec[i], _RobotRVec[i + 1], _RobotRVec[i + 2], mat, MJROTATE_ZXZ);
		RobotRMat.push_back(mat);
		RobotTMat.push_back(transmat);
	}

	for (int j = 0; j < _CamRVec.size(); j += 3)
	{
		cv::Vec3d CamRVec = { _CamRVec[j],_CamRVec[j + 1],_CamRVec[j + 2] };
		cv::Mat mat = cv::Mat(3, 3, CV_64FC1);
		cv::Mat transmat = (cv::Mat_<double>(3, 1) << _CamTVec[j], _CamTVec[j + 1], _CamTVec[j + 2]);
		cv::Rodrigues(CamRVec, mat);
		CamRMat.push_back(mat);
		CamTMat.push_back(transmat);
	}

	cv::calibrateHandEye(RobotRMat, RobotTMat, CamRMat, CamTMat, _Cam2EndR, _Cam2EndT, cv::CALIB_HAND_EYE_TSAI);
}

void MCaliHandEye::XYZ2TransMat(double _DeltaX, double _DeltaY, double _DeltaZ, cv::Mat _TransMat)
{
	_TransMat = (cv::Mat_<double>(3, 1) << _DeltaX, _DeltaY, _DeltaZ);
}

void MCaliHandEye::Euler2RotateMat(double _DegreeA, double _DegreeB, double _DegreeC, cv::Mat& _RotateMat, int _iType)
{
	if (_iType == MJROTATE_ZXZ)
	{
		_RotateMat = RotateZMat(_DegreeA) * RotateXMat(_DegreeB) * RotateZMat(_DegreeC);
	}
	else if (_iType == MJROTATE_ZYX)
	{
		_RotateMat = RotateZMat(_DegreeC) * RotateYMat(_DegreeB) * RotateXMat(_DegreeA);
	}
}

cv::Mat MCaliHandEye::RotateXMat(double _Degree)
{
	_Degree /= (180 / CV_PI);
	return (cv::Mat_<double>(3, 3) << 1, 0, 0,
		0, cos(_Degree), -(sin(_Degree)),
		0, sin(_Degree), cos(_Degree));
}

cv::Mat MCaliHandEye::RotateYMat(double _Degree)
{
	_Degree /= (180 / CV_PI);

	return (cv::Mat_<double>(3, 3) << cos(_Degree), 0, sin(_Degree),
		0, 1, 0,
		-sin(_Degree), 0, cos(_Degree));
}

cv::Mat MCaliHandEye::RotateZMat(double _Degree)
{
	_Degree /= (180 / CV_PI);

	return (cv::Mat_<double>(3, 3) << cos(_Degree), -sin(_Degree), 0,
		sin(_Degree), cos(_Degree), 0,
		0, 0, 1);
}


