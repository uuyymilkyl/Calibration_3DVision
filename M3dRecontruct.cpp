#include "M3dRecontruct.hpp"
#include <iostream>
#include <fstream>
#include <iostream>
using namespace std;



M3DRecontruct::M3DRecontruct()
{ 
	std::string CamCaliParam = "./CaliResult/CameraCalibration.param";
	std::string LaserCaliParam = "./CaliResult/LaserCalibration.param";
	std::string HandEyeCaliParam = "./CaliResult/HandEyeCalibration.param";

	std::string FramePath = "./data/*.jpg";
	std::string ArmPoseTxtPath = "./newdata.txt";


	LoadCalibraionParam(CamCaliParam,LaserCaliParam,HandEyeCaliParam);
	//实际位置 = T点到相机的位置 + T相机到手的位置 + T手到基座的位置
	//实际角度 = 点云拼接后两个夹角面的均角与Z轴的夹角
	std::vector<cv::Mat> ArmPosePoint = LoadArmPoseByTxt(ArmPoseTxtPath);

	TestRecontructByFrame(FramePath, ArmPoseTxtPath);
	

}

M3DRecontruct::~M3DRecontruct()
{
}

void M3DRecontruct::WritedownCloudPoints(std::vector<cv::Point3f>& _vec_ptDst, std::string _name)
{

	std::ofstream file(_name);
	if (file.is_open()) {
		file << "# .PCD v0.7 - Point Cloud Data file format\n";
		file << "VERSION 0.7\n";
		file << "FIELDS x y z\n";
		file << "SIZE 4 4 4\n";
		file << "TYPE F F F\n";
		file << "COUNT 1 1 1\n";
		file << "WIDTH " << _vec_ptDst.size() << "\n";
		file << "HEIGHT 1\n";
		file << "VIEWPOINT 0 0 0 1 0 0 0\n";
		file << "POINTS " << _vec_ptDst.size() << "\n";
		file << "DATA ascii\n";

		for (const auto& point : _vec_ptDst) {
			file << point.x << " " << point.y << " " << point.z << "\n";
		}
		file.close();
		std::cout << "Point cloud data has been successfully written to " << std::endl;
	}
	else {
		std::cerr << "Unable to open for writing" << std::endl;
	}

}

void M3DRecontruct::LoadCalibraionParam(std::string& _CamCaliXmlDir, std::string& _LaserCaliXmlDir, std::string& _HandEyeCaliXmlDir)
{

	//加载相机标定
	cv::Mat IntrinsicMatrix(3, 3, CV_64F);
	cv::Mat DistortionCoeffs(5, 1, CV_64F);
	
	
	std::ifstream CamCaliXml;
	CamCaliXml.open(_CamCaliXmlDir);
	if (!CamCaliXml.is_open())
	{
		std::cerr << "Error opening file: " << _CamCaliXmlDir << std::endl;
	}

	std::string CamCaliLine;
	int i = 0;
	while (std::getline(CamCaliXml, CamCaliLine))
	{
		std::istringstream iss(CamCaliLine);
		if (i >= 1 && i <= 3)
		{
			double a, b, c;
			iss >> a >> b >> c;
			IntrinsicMatrix.at<double>(i-1, 0) = a;
			IntrinsicMatrix.at<double>(i-1, 1) = b;
			IntrinsicMatrix.at<double>(i-1, 2) = c;
				
		}
		if (i == 5)
		{
			for (int j = 0; j < 5; j++)
			{
				double a ;
				iss >> a ;
				DistortionCoeffs.at<double>(j, 0) = a; // 将数据放入矩阵
			}

		}
		i += 1;
	}
	


	
	//加载光平面标定

	cv::Mat LaserPlaneCoeffs(4, 1, CV_64F);
	std::ifstream LaserCaliXml;
	LaserCaliXml.open(_LaserCaliXmlDir);
	if (!LaserCaliXml.is_open())
	{
		std::cerr << "Error opening file: " << _LaserCaliXmlDir << std::endl;
	}
	std::string LaserCaliLine;
	int z = 0;
	while (std::getline(LaserCaliXml, LaserCaliLine))
	{
		std::istringstream iss(LaserCaliLine);
		if (z == 1)
		{
			for (int i = 0; i < 4; ++i) //遍历4个数
			{

				double value;
				iss >> value;
				LaserPlaneCoeffs.at<double>(i, 0) = value; // 将数据放入矩阵
			}
			
		}
		z += 1;
	}


	//加载手眼标定
	cv::Mat R_HandEyeMatrix(3, 3, CV_64F);
	cv::Mat T_HandEyeMatrix(3, 1, CV_64F);
	std::ifstream HandEyeCaliXml;
	HandEyeCaliXml.open(_HandEyeCaliXmlDir);
	if (!HandEyeCaliXml.is_open())
	{
		std::cerr << "Error opening file: " << _HandEyeCaliXmlDir << std::endl;
	}
	std::string HandEyeCali_Line;       //声明string

	while (std::getline(HandEyeCaliXml, HandEyeCali_Line))  //getline 
	{
		std::istringstream iss(HandEyeCali_Line);



		if (HandEyeCali_Line == "[HandEyeMatrix_R]") {
			for (int i = 0; i < 3; ++i)
			{
				std::getline(HandEyeCaliXml, HandEyeCali_Line);
				std::istringstream iss(HandEyeCali_Line);

				double a, b, c;
				iss >> a >> b >> c;
				R_HandEyeMatrix.at<double>(i, 0) = a;
				R_HandEyeMatrix.at<double>(i, 1) = b;
				R_HandEyeMatrix.at<double>(i, 2) = c;
			}
		}
		else if (HandEyeCali_Line == "[HandEyeMatrix_T]")
		{
			for (int i = 0; i < 3; ++i)
			{
				std::getline(HandEyeCaliXml, HandEyeCali_Line);
				std::istringstream iss(HandEyeCali_Line);

				double val;
				iss >> val;
				T_HandEyeMatrix.at<double>(i, 0) = val;
			}
		}
	}

	this->m_IntrinxMatrix = IntrinsicMatrix;
	this->m_DistortCoeffs = DistortionCoeffs;
	this->m_LaserPlaneCoeffs = LaserPlaneCoeffs;
	this->m_HandEyeMatrix_R = R_HandEyeMatrix;
	this->m_HandEyeMatrix_T = T_HandEyeMatrix;
	
}



void M3DRecontruct::VideoRecontruct(std::string& _videopath)
{
	// 打开视频文件
	cv::VideoCapture cap(_videopath);

	cv::namedWindow("Video", cv::WINDOW_NORMAL);
	std::vector<cv::Point3f> LaserLinesPoints;
	int framecount = -50;
	while (true)
	{
		float vframey = 0.42;
		float vframex = -0.1;

		// 读取第一帧
		cv::Mat frame;

		if (!cap.read(frame)) {
			break; // Break the loop if no frame is retrieved
		}

		//cv::imshow("Video", frame);
		//cv::waitKey(2);

		int rowa = 3;
		int cola = 3;
		double IntrinxMatData[] = {
			1267.585781118558, 0, 571.9786672135109,
			0, 1268.559400548863, 371.0975948878292,
			0, 0, 1 };

		cv::Mat IntrinxMatrix(rowa, cola, CV_64F, IntrinxMatData);
		double u = IntrinxMatrix.at<double>(0, 2);
		double v = IntrinxMatrix.at<double>(1, 2);

		int rowb = 5;
		int colb = 1;
		double DistortCoeffsData[] = {
			-0.2283319097884629,-0.1125851879085977,0.0002994858863522342,-0.0003438434346782177, 0.08551944660237307 };
		cv::Mat	DistortCoeMatrix(rowb, colb, CV_64F, DistortCoeffsData);


		int rowc = 4;
		int colc = 1;
		double LaserPlaneData[] = {
			-0.394446 ,-0.857732 ,-0.32971 ,-207.501 };
		cv::Mat	 LaserPlaneMatrix(rowc, colc, CV_64F, LaserPlaneData);

		std::vector<cv::Point2f> testLaserPoints;
		std::vector<cv::Point2f> testLaserPoints_sobel;
		std::vector<cv::Point3f> dstPoints;
	
		//通过红色提取激光点
		ExtractRedLaserContour(frame, testLaserPoints_sobel);

		cv::Mat showimg = frame.clone();

		for (int i = 0; i < testLaserPoints_sobel.size(); i++)
		{
			//cv::circle(showimg, testLaserPoints_sobel[i], 1, cv::Scalar(0, 255, 0), 2, 6, 0);

		}

		MCalibLaser::ImgPtsToCamFrame(LaserPlaneMatrix, IntrinxMatrix, DistortCoeMatrix, testLaserPoints_sobel, dstPoints);
		for (int i = 0; i < dstPoints.size(); i++)
		{
			dstPoints[i].y -= (framecount * 3 * vframey);
			dstPoints[i].y += (120 - i) * 0.05;
			dstPoints[i].x += (80 - framecount) * 0.01;
			LaserLinesPoints.push_back(dstPoints[i]);

		}

		framecount += 1;
		//MCalibLaser::Writedown(dstPoints, "test1.pcd");
	}

	MCalibLaser::Writedown(LaserLinesPoints, "showban.pcd");
	// 释放资源
	cap.release();
	cv::destroyAllWindows();
}

void M3DRecontruct::TestRecontructByFrame(std::string& _ImagePath, std::string& _ArmPosePath)
{



	std::string FramePath = _ImagePath;
	std::vector<cv::String> FrameDirList;

	std::string TxtPath = _ArmPosePath;
	std::vector<cv::Mat> FrameArmPoseList;


	cv::glob(FramePath, FrameDirList);
	FrameArmPoseList = LoadArmPoseByTxt(TxtPath);

	std::vector<cv::Point3f> ConnectedCloudPoints_150;
	std::vector<cv::Point3f> ConnectedCloudPoints_a;
	for (int i = 0; i < 200; i++)
	{
		cv::Mat nFrame = cv::imread(FrameDirList[i]);
		cv::Mat nArmPose = FrameArmPoseList[i];
		ConnectedCloudPoints_a = AccumulateFrameRecontruct(nFrame, m_HandEyeMatrix_R, m_HandEyeMatrix_T, nArmPose);
		ConnectedCloudPoints_150.insert(ConnectedCloudPoints_150.end(), ConnectedCloudPoints_a.begin(), ConnectedCloudPoints_a.end());
		ConnectedCloudPoints_a.clear();
	}

	WritedownCloudPoints(ConnectedCloudPoints_150, "Test15.pcd");

	//实际位置 = T点到相机的位置 + T相机到手的位置 + T手到基座的位置
	//实际角度 = 点云拼接后两个夹角面的均角与Z轴的夹角
}

std::vector<cv::Point3f> M3DRecontruct::AccumulateFrameRecontruct(cv::Mat& _frame, cv::Mat& _HandEyeMatrix_R, cv::Mat& _HandEyeMatrix_T, cv::Mat& _CurrenArmPose)
{
	//单帧通过
	std::vector<cv::Point3f> CloudPointsOfAPlane;  ///< 声明单帧点云结果（相机坐标系下）

	cv::Mat frame = _frame;                        ///< 获取本帧图像

	//cv::imshow("Video", frame);
	//cv::waitKey(2);

	cv::Mat IntrinxMatrix =  this->m_IntrinxMatrix;      ///< 获取相机内参

	cv::Mat	DistortCoeMatrix = this->m_DistortCoeffs;    ///< 获取畸变系数

	cv::Mat HandEyeCoeMatrix_R = this->m_HandEyeMatrix_R;

	cv::Mat HandEyeCoeMatrix_T = this->m_HandEyeMatrix_T;

	cv::Mat	 LaserPlaneMatrix = this->m_LaserPlaneCoeffs;///< 获取本帧图像

	//断点1 -------  ↑ 检查加载的参数

	std::vector<cv::Point2f> testLaserPoints;
	std::vector<cv::Point2f> testLaserPoints_sobel;
	std::vector<cv::Point3f> dstPoints;
		
	ExtractRedLaserContour(frame, testLaserPoints_sobel);

	cv::Mat showimg = frame.clone();

	for (int i = 0; i < testLaserPoints_sobel.size(); i++)
	{
		cv::circle(showimg, testLaserPoints_sobel[i], 1, cv::Scalar(0, 255, 0), 2, 6, 0);
	}
	

	MCalibLaser::ImgPtsToCamFrame(LaserPlaneMatrix, IntrinxMatrix, DistortCoeMatrix, testLaserPoints_sobel, dstPoints);

	//WritedownCloudPoints(dstPoints, "CameraPoints.pcd");
	
	//dstPoint
	cv::Mat T_extended = cv::Mat::ones(4, 1, CV_64F);
	m_HandEyeMatrix_T.copyTo(T_extended(cv::Rect(0, 0, 1, 3)));

	// 将R和T拼接成一个3x4的矩阵
	cv::Mat RT;

	cv::Mat Single_R = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	//cv::hconcat(m_HandEyeMatrix_R, m_HandEyeMatrix_T, RT);
	hconcat(m_HandEyeMatrix_R, m_HandEyeMatrix_T, RT);

	//cv::Mat ArmPose_R = _CurrenArmPose.colRange(3, 6).clone();

	cv::Mat lastRow = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);

	// 将RT和新的一行拼接成一个4x4的矩阵
	cv::Mat HandEyeMatrix;
	cv::vconcat(RT, lastRow, HandEyeMatrix);

	// 求逆（没用到）
	cv::Mat Inv_HandEyeMatrix;
	cv::invert(HandEyeMatrix, Inv_HandEyeMatrix);

	// 计算机械臂矩阵
	cv::Mat ArmPose_T = (cv::Mat_<double>(4, 1) << _CurrenArmPose.at<double>(0, 0),
		_CurrenArmPose.at<double>(0, 1),
		_CurrenArmPose.at<double>(0, 2),
		1);

	cv::Mat Ra = _CurrenArmPose.colRange(3, 6).clone();

	cv::Mat ArmPose_R = (cv::Mat_<double>(1,3)<< 0 ,0 ,0);

	cv::Mat ArmPoseMatrixRT = buildPoseMatrix(ArmPose_T, Ra);

	cv::Mat ArmPose_R_3x3 = eulerToRotationMatrix(Ra);
	
	for (int i = 0; i < dstPoints.size(); i++)
	{
		cv::Mat CameraPoint = (cv::Mat_<double>(4, 1) << dstPoints[i].x, dstPoints[i].y, dstPoints[i].z, 1);
		cv::Mat HandPoint = ArmPoseMatrixRT * HandEyeMatrix * CameraPoint ;

		cv::Mat RotateMat_3x3 = ArmPose_R_3x3 * m_HandEyeMatrix_R;

		cv::Mat WorldRotateMat_4x4 = (cv::Mat_<double>(4, 4) << RotateMat_3x3.at<double>(0, 0), RotateMat_3x3.at<double>(0, 1), RotateMat_3x3.at<double>(0, 2),0,
			                                               RotateMat_3x3.at<double>(1, 0), RotateMat_3x3.at<double>(1, 1), RotateMat_3x3.at<double>(1, 2),0,
			                                               RotateMat_3x3.at<double>(2, 0), RotateMat_3x3.at<double>(2, 1), RotateMat_3x3.at<double>(2, 2),0,
			                                                0,0,0,1);
		cv::Mat WorldPoint =  HandPoint;

		cv::Point3f WorldPoint3f;
		WorldPoint3f.x = WorldPoint.at<double>(0, 0);
		WorldPoint3f.y = WorldPoint.at<double>(1, 0);
		WorldPoint3f.z = WorldPoint.at<double>(2, 0);
		//与上一帧的位置偏移
		CloudPointsOfAPlane.push_back(WorldPoint3f);
	}

	WritedownCloudPoints(dstPoints, "Camera.pcd");
	

	return CloudPointsOfAPlane;
}

cv::Mat M3DRecontruct::eulerToRotationMatrix(const cv::Mat& angles)
{
	double pi_180 = 3.1415926535/180.0;

	double alpha = angles.at<double>(0) *pi_180;
	double beta = angles.at<double>(1) * pi_180;
	double gamma = angles.at<double>(2) * pi_180;

	// ZXZ Euler angles to rotation matrix
	cv::Mat R_xz = (cv::Mat_<double>(3, 3) 
		<< cos(alpha), -sin(alpha), 0,
		sin(alpha), cos(alpha), 0,
		0, 0, 1);

	cv::Mat R_xx = (cv::Mat_<double>(3, 3) 
		<< 1, 0, 0,
		0, cos(beta), -sin(beta),
		0, sin(beta), cos(beta));

	cv::Mat R_zz = (cv::Mat_<double>(3, 3)
		<< cos(gamma), -sin(gamma), 0,
		sin(gamma), cos(gamma), 0,
		0, 0, 1);

	// Combine the rotations
	cv::Mat R = R_xz * R_xx * R_zz  ;
	return R;


}

cv::Mat M3DRecontruct::buildPoseMatrix(const cv::Mat& t, const cv::Mat& angles)
{
	cv::Mat R = eulerToRotationMatrix(angles);

	// Build pose matrix
	cv::Mat pose = cv::Mat::eye(4, 4, R.type()); // 创建一个4x4的单位矩阵
	R.copyTo(pose(cv::Rect(0, 0, 3, 3))); // 将R复制到pose的左上角3x3位置
	t.copyTo(pose(cv::Rect(3, 0, 1, 4))); // 将t复制到pose的右侧3x1位置

	return pose;
}

std::vector<cv::Mat> M3DRecontruct::LoadArmPoseByTxt(std::string &_Dir)
{
	std::vector<cv::Mat> ArmPoseList;
	

	std::ifstream file(_Dir);
	std::string ArmPoseLine;
	// 这里不需要用i来挑行数，因此用while即可
	while (std::getline(file, ArmPoseLine)) 
	{
		std::istringstream iss(ArmPoseLine);
		cv::Mat ArmPoseEach(1, 6, CV_64F);
		for (int j = 0; j < 6; j++)
		{
			double value;
			iss >> value;
			ArmPoseEach.at<double>(0, j) = value;
		}
		ArmPoseList.push_back(ArmPoseEach);
		
	}


	return ArmPoseList;
}
