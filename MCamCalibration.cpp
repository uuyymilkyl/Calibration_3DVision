
#include"MCamCalibration.hpp"

MCalibCam::MCalibCam()
{
	std::string ImgDir = "./CaliCamIn/*.jpg";

	//加载棋盘格数据 横向内角点数，纵向内角点数，棋盘格间隔（mm)
	SetBoardSize(11, 8, 5);

	//加载图像和棋盘格数据
	LoadChessBoardData(ImgDir);

	//设置像元尺寸（镜头出厂数据）
	this->m_PixelSize = 0.003; //0.003mm - 3微米

	// 找到棋盘格角点1
	std::vector<std::vector<cv::Point2f>> imagePoints;

	// 声明内参矩阵2，畸变矩阵3
	cv::Mat intrinxMatrix, coeffsMatrix;

	// 标定出123
	CalibrationOfCamera(intrinxMatrix, coeffsMatrix);

	std::vector<cv::Mat> Rvecs, Tvecs; //每幅图像的旋转矩阵4 平移矩阵5

	calibrateCamera(this->m_objectPoints, this->m_imagePoints, m_ImgSize, intrinxMatrix, coeffsMatrix, Rvecs, Tvecs);


	// 打印标定结果
	std::cout << "Camera Matrix:\n" << intrinxMatrix << "\n\n";
	std::cout << "Distortion Coefficients:\n" << coeffsMatrix << "\n\n";

	m_IntrinxMatrix = intrinxMatrix;
	m_CoeffsMatrix = coeffsMatrix;


	/*
	for (int i = 0; i < m_ImgList.size(); i++)
	{
		// 每一张图计算所有角点的距离
		//CalculateEvConerDistance(m_imagePoints[i], i,Rvecs[i],Tvecs[i]);

	}
	*/

}

MCalibCam::~MCalibCam()
{
}
 // 根据内参矩阵K和畸变系数distCoeffs反投影图像上的点
cv::Point2f undistortPoint(const cv::Point2f& distorted_point, const cv::Mat& K, const cv::Mat& distCoeffs) {
	cv::Mat distorted = (cv::Mat_<float>(1, 1) << distorted_point.x, distorted_point.y);
	cv::Mat undistorted;
	cv::undistortPoints(distorted, undistorted, K, distCoeffs);
	return cv::Point2f(undistorted.at<float>(0, 0), undistorted.at<float>(0, 1));
}

void MCalibCam::SetBoardSize(int _nWidthLatticeNum, int _nHeightLatticeNum, float _fSquarSize)
{
	m_HeightLatticeNum = _nHeightLatticeNum;
	m_WidthLattinceNum = _nWidthLatticeNum;
	m_SquarSize = _fSquarSize;
}

void MCalibCam::LoadChessBoardData(std::string& _ImgDir)
{
	std::vector<cv::String> ImgList;
	cv::glob(_ImgDir, ImgList);
	m_ImgList = ImgList;


	std::vector<std::vector<cv::Point3f>> objectPoints;
	std::vector<cv::Point3f> realChessboardData;

	for (int i = 0; i < m_HeightLatticeNum; ++i)
	{
		for (int j = 0; j < m_WidthLattinceNum; ++j)
		{
			realChessboardData.push_back(cv::Point3f(j * m_SquarSize, i * m_SquarSize, 0.0f));
		}
	}

	for (size_t i = 0; i < ImgList.size(); ++i)
	{
		objectPoints.push_back(realChessboardData);
	}
	m_objectPoints = objectPoints;
}

void MCalibCam::CalibrationOfCamera(cv::Mat& _intrinxMatrix, cv::Mat& _coeffsMatrix)
{
	std::vector<cv::Point2f> corners;

	for (int i = 0; i < m_ImgList.size(); ++i)
	{
		cv::Mat image = cv::imread(m_ImgList[i]);
		cvtColor(image, image, cv::COLOR_BGR2GRAY);

		//找棋盘格角点
		bool found = findChessboardCorners(image, cv::Size(m_WidthLattinceNum, m_HeightLatticeNum), corners);

		if (found)
		{
			// 亚像素定位精确找寻角点 
			cornerSubPix(image, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
			// 将本图中找到的角点集放回vector中
			m_imagePoints.push_back(corners);
			// 画出角点（可视化操作）
			drawChessboardCorners(image, cv::Size(m_WidthLattinceNum, m_HeightLatticeNum), corners, found);
		}

		m_ImgSize = image.size();
		//m_imagePoints.push_back(corners);
	}

}

void MCalibCam::CalEConerCoordinationInCamera(std::vector<cv::Point2f>_corner,int imgindex,cv::Mat Rvecs, cv::Mat Tvecs)
{
	double fx = m_IntrinxMatrix.at<double>(0, 0);  // 焦距在x轴上的分量
	double fy = m_IntrinxMatrix.at<double>(1, 1);  // 焦距在y轴上的分量
	double cx = m_IntrinxMatrix.at<double>(0, 2);  // 图像中心的x坐标
	double cy = m_IntrinxMatrix.at<double>(1, 2);  // 图像中心的y坐标
	double PixelSize = 0.003;


	cv::Mat rotationMatrix;
	cv::Mat imagePoint;
	for (int i = 0; i < _corner.size(); i++)
	{
		imagePoint = (cv::Mat_<double>(3, 1) << double(_corner[i].x), double(_corner[i].y), 1);
		//计算比例参数S

		double s;
		cv::Mat tempMat, tempMat2;
		cv::Mat rotationMatrix;
		Rodrigues(Rvecs, rotationMatrix);

		tempMat = rotationMatrix.inv() * m_IntrinxMatrix.inv() * imagePoint;
		tempMat2 = rotationMatrix.inv() * Tvecs;
		s = tempMat2.at<double>(2, 0);
		s /= tempMat.at<double>(2, 0);

		// 计算世界坐标
		cv::Mat matWorldPoints = rotationMatrix.inv() * (s * m_IntrinxMatrix.inv() * imagePoint - Tvecs);

		// 计算相机坐标
		cv::Mat matCameraPoints = rotationMatrix * matWorldPoints + Tvecs;


		cv::Point3f ptCameraPoints(matCameraPoints.at<double>(0, 0), matCameraPoints.at<double>(1, 0), matCameraPoints.at<double>(2, 0));


	}
	
}

cv::Point3f MCalibCam::CalPlaneLineIntersectPoint(const cv::Vec3d& planeNormal, const cv::Point3f& planePoint, const cv::Vec3f& lineDirection, const cv::Point3f& linePoint)
{
	double denom = planeNormal.dot(lineDirection);
	if (std::abs(denom) < 1e-6) 
{
		// 平行或重合，没有交点
		return cv::Point3f(NAN, NAN, NAN); // 返回无效的点
	}
	else {
		cv::Vec3f diff = cv::Vec3f(planePoint.x - linePoint.x, planePoint.y - linePoint.y, planePoint.z - linePoint.z);
		double t = diff.dot(planeNormal) / denom;
		cv::Point3f intersectionPoint(linePoint.x + t * lineDirection[0],
			linePoint.y + t * lineDirection[1],
			linePoint.z + t * lineDirection[2]);
		return intersectionPoint;
	}
}


