#include "MLasCalibration.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <fstream>

float calculateDistance(cv::Point2f p1, cv::Point2f p2) 
{
	return std::sqrt(std::pow(p2.x - p1.x, 2) + std::pow(p2.y - p1.y, 2));
}
std::vector<cv::Point2f> GetLinePointsGrayWeight(cv::Mat& src, int gray_Thed, int Min, int Max, int Type)
{
	std::vector<cv::Point2f> points_vec;
	if (Type == 0)
	{
		Min = Min < 0 ? 0 : Min;
		Max = Max > src.rows ? src.rows : Max;
		for (int i = 0; i < src.cols; i++)
		{
			float X0 = 0, Y0 = 0;
			for (int j = Min; j < Max; j++)
			{
				if (src.at<ushort>(j, i) > gray_Thed)
				{
					X0 += src.at<ushort>(j, i) * j;
					Y0 += src.at<ushort>(j, i);
				}
			}
			if (Y0 != 0)
			{
				//Point p = Point(i, X0 / Y0);
				points_vec.push_back(cv::Point2f(i, X0 / (float)Y0));
			}
			else
			{
				//points_vec.push_back(Point2f(i, -1));
			}
		}
	}
	else
	{
		Min = Min < 0 ? 0 : Min;
		Max = Max > src.cols ? src.cols : Max;
		for (int i = 0; i < src.rows; i++)
		{
			int X0 = 0, Y0 = 0;
			for (int j = Min; j < Max; j++)
			{
				if (src.at<cv::Vec3b>(i, j)[0] > gray_Thed)
				{
					X0 += src.at<cv::Vec3b>(i, j)[0] * j;
					Y0 += src.at<cv::Vec3b>(i, j)[0];
				}
			}
			if (Y0 != 0)
			{
				points_vec.push_back(cv::Point2f(X0 / (float)Y0, i));
			}
			else
			{
				points_vec.push_back(cv::Point2f(-1, i));
			}
		}
	}
	return points_vec;
}

void FitPlaneRANSAC(std::vector<cv::Point3f> _vecPoints3ds, cv::Mat& _mat_plane) {
	int max_iterations = 1000; // 最大迭代次数
	double threshold_distance = 0.01; // 阈值距离
	int inliers_required = _vecPoints3ds.size() * 0.7; // 所需内点数量

	if (_vecPoints3ds.size() < 3) {
		// 如果输入的点云数量小于3，无法拟合平面
		std::cerr << "Error: Insufficient points for plane fitting" << std::endl;
		return;
	}

	cv::RNG rng; // 随机数生成器
	cv::Mat best_plane; // 保存最佳平面参数
	int best_inliers = 0; // 最佳内点数量

	for (int i = 0; i < max_iterations; ++i) {
		// 随机选择三个点
		int idx1 = rng.uniform(0, _vecPoints3ds.size());
		int idx2 = rng.uniform(0, _vecPoints3ds.size());
		int idx3 = rng.uniform(0, _vecPoints3ds.size());

		// 确保三个索引不相同
		if (idx1 == idx2 || idx1 == idx3 || idx2 == idx3) {
			continue;
		}

		// 构建平面模型
		cv::Point3f p1 = _vecPoints3ds[idx1];
		cv::Point3f p2 = _vecPoints3ds[idx2];
		cv::Point3f p3 = _vecPoints3ds[idx3];

		cv::Point3f v1 = p2 - p1;
		cv::Point3f v2 = p3 - p1;

		cv::Point3f normal = v1.cross(v2);
		normal /= cv::norm(normal);

		double d = -normal.dot(p1);

		// 计算内点数量
		int inliers = 0;
		for (const auto& pt : _vecPoints3ds) {
			double distance = std::abs(normal.dot(pt) + d);
			if (distance < threshold_distance) {
				inliers++;
			}
		}

		// 更新最佳平面参数和内点数量
		if (inliers > best_inliers) {
			best_inliers = inliers;
			best_plane = (cv::Mat_<double>(1, 4) << normal.x, normal.y, normal.z, d);
		}

		// 如果找到足够多的内点，提前结束循环
		if (inliers >= inliers_required) {
			break;
		}
	}

	// 将最佳平面参数保存到输出参数中
	_mat_plane = cv::Mat::zeros(1, 4, CV_64FC1);
	_mat_plane = best_plane;

}

void fitPlane(const std::vector<cv::Point3f>& points, float* planeCoefficients) {
	// 计算点的重心
	cv::Point3f centroid(0, 0, 0);
	for (const auto& point : points) {
		centroid += point;
	}
	centroid *= (1.0 / points.size());

	// 构建协方差矩阵
	cv::Mat covariance_matrix = cv::Mat::zeros(3, 3, CV_32F);
	for (const auto& point : points) {
		cv::Mat deviation = (cv::Mat_<float>(3, 1) << point.x - centroid.x, point.y - centroid.y, point.z - centroid.z);
		covariance_matrix += deviation * deviation.t();
	}
	covariance_matrix /= static_cast<float>(points.size());

	// 计算协方差矩阵的特征值和特征向量
	cv::Mat eigenvalues, eigenvectors;
	cv::eigen(covariance_matrix, eigenvalues, eigenvectors);

	// 获取最小特征值对应的特征向量作为平面的法向量
	cv::Mat normal = eigenvectors.colRange(0, 1).clone();

	// 设置平面参数
	planeCoefficients[0] = normal.at<float>(0, 0);
	planeCoefficients[1] = normal.at<float>(1, 0);
	planeCoefficients[2] = normal.at<float>(2, 0);
	planeCoefficients[3] = -normal.dot(cv::Mat(centroid));
}

void getLineStartEndPoints(const cv::Vec4f& line, float xmin, float xmax, float ymin, float ymax, cv::Point& startPoint, cv::Point& endPoint) {
	if (line[1] != line[3]) {
		startPoint.x = ((ymin - line[1]) * (line[2] - line[0])) / (line[3] - line[1]) + line[0];
		endPoint.x = ((ymax - line[1]) * (line[2] - line[0])) / (line[3] - line[1]) + line[0];
		startPoint.y = ymin;
		endPoint.y = ymax;
	}
	else {
		startPoint.y = ((xmin - line[0]) * (line[3] - line[1])) / (line[2] - line[0]) + line[1];
		endPoint.y = ((xmax - line[0]) * (line[3] - line[1])) / (line[2] - line[0]) + line[1];
		startPoint.x = xmin;
		endPoint.x = xmax;
	}
}


// 计算三维点集到空间平面的平均距离
float calPlaneDist(const std::vector<cv::Point3f>& points, float _a,float _b, float _c, float _d) {
	float totalDistance = 0.0f;
	int numPoints = static_cast<int>(points.size());

	// 平面方程系数
	float a = _a;
	float b = _b;
	float c = _c;
	float d = _d;

	// 计算每个点到平面的距离
	for (const auto& point : points) {
		float x = point.x;
		float y = point.y;
		float z = point.z;

		// 计算点到平面的距禿
		float distance = std::abs(a * x + b * y + c * z + d) / std::sqrt(a * a + b * b + c * c);
		totalDistance += distance;
	}

	// 计算平均距离
	float averageDistance = totalDistance / static_cast<float>(numPoints);
	return averageDistance;
}

MCalibLaser::MCalibLaser()
{
	

	int rowa = 3;
	int cola = 3;
	double IntrinxMatData[] = { 
		1554.877403928442, 0, 619.8165844670023,
		0, 1555.687350274757, 371.0289583775858,
		0, 0, 1 };

	cv::Mat IntrinxMatrix(rowa, cola, CV_64F, IntrinxMatData);
	double u = IntrinxMatrix.at<double>(0, 2);
	double v = IntrinxMatrix.at<double>(1, 2);

	int rowb = 5;
	int colb = 1;
	double DistortCoeffsData[] = {
		-0.2429598680402254, -0.35631626581568,-0.001725313674369029,-0.00598359825434598,- 0.1013544978183295 };

	cv::Mat DistortCoeffs(rowb, colb, CV_64F, DistortCoeffsData);

	std::string ImgDir = "./CaliLaser/CaliLaser1202/*.jpg";
	std::vector<cv::Mat> CaliCamImg;
	std::vector<cv::Mat> CaliLasImg;

	LoadLaserImage(ImgDir, CaliCamImg, CaliLasImg,IntrinxMatrix,DistortCoeffs);

	//获取每张标定板图的外参
	std::vector<cv::Mat> Rcves, Tcves;
	CalibraCamForLaserImg(CaliCamImg, Rcves, Tcves);
	
	std::vector<cv::Point2f> vLaserPoints;
	std::vector<cv::Point2f> vFilterLaserPoints;
	std::vector<cv::Point3f> vPlane_Points_Vec;
	std::vector<cv::Point3f> vPlane_Points_Vec2;
	std::vector<std::vector<cv::Point3f>> vvPlane_Points_Vec2;
	for (int i = 0; i < CaliLasImg.size(); i++)
	{
		cv::Mat showimg = CaliLasImg[i].clone();
		//每张图求得标定板上的直线
		std::vector<cv::Vec4f> ChessBoardLines;
		std::vector<std::vector<cv::Point2f>> ChessLinesPoints;
		ChessBoardLines = FitChessBoardLines(this->m_imagePoints[i], CaliCamImg[i],ChessLinesPoints);

		//求激光拟合的轮廓点
		//vLaserPoints = GetLaserPoints_Steger(CaliLasImg[i], 190, 0, 1280, 1);

		ExtractRedLaserContour(CaliLasImg[i], vLaserPoints);
		//用棋盘格筛选激光轮廓点
		int xmin;
		int xmax;
		int ymin;
		int ymax;
		GetCaliRangeAndFitLine(CaliCamImg[i], xmin, ymin, xmax, ymax);
		vFilterLaserPoints = FilterLaserPointsByRange(vLaserPoints, xmin, ymin, xmax, ymax);

		for (int i = 0; i < vFilterLaserPoints.size(); i++)
		{
			cv::circle(showimg, vFilterLaserPoints[i], 1, cv::Scalar(255, 0, 0), 2, 8, 0);
		}

		//拟合成直线
		cv::Vec4f LaserFitLine;
		cv::fitLine(vFilterLaserPoints, LaserFitLine, cv::DIST_L2, 0, 0.01, 0.01);

		cv::Point startp;
		cv::Point endp;
		getLineStartEndPoints(LaserFitLine, xmin, xmax, ymin, ymax, startp, endp);

		float k = LaserFitLine[1] / LaserFitLine[0]; // 斜率
		float b = LaserFitLine[3] - k * LaserFitLine[2]; // 截距
		cv::line(showimg, cv::Point(0, b - 1), cv::Point(showimg.cols + 1, k * showimg.cols + b), cv::Scalar(0, 255, 0), 2);


		//求交点
		std::vector<cv::Point2f> cross_vec;
		cv::Mat showcrossimg = CaliLasImg[i].clone();
		for (cv::Vec4f nChessline : ChessBoardLines)
		{
			/*求交点并画点保存，result.jpg存储在工程目录下*/
			cv::Point2f crossPoint;
			crossPoint = getCrossPoint(nChessline, LaserFitLine);
			cross_vec.push_back(crossPoint);
			circle(showcrossimg, crossPoint, 3, cv::Scalar(200, 0, 20), 3, 8, 0);
		}
		
		//激光线标板坐标系坐标提取（交比不变性）
		//根据标板坐标系和图像坐标系的交并复比不变性
		//求取交点在靶标坐标系上的点坐标
		std::vector<cv::Point3f> Che_Points;   //标板坐标系激光点
		std::vector<cv::Point3f> Cam_Points;   //相机坐标系激光点
		int Chess_Size = 3;

		std::vector<cv::Point2f> cross_vec_undisort;
		DistortPoints(cross_vec, cross_vec_undisort, IntrinxMatrix, DistortCoeffs);  //矫正激光标版交点

		for (int m = 0; m < ChessLinesPoints.size(); m++)
		{
			
			std::vector<cv::Point2f> Chess_points;
			DistortPoints( ChessLinesPoints[m], Chess_points, IntrinxMatrix, DistortCoeffs);   //图像坐标系：标板直线角点,矫正标版角点
			
			//图像坐标系的交并复比
			/*
			double AC = calculateDistance(Chess_points[10], Chess_points[5]);
			double AD = calculateDistance(Chess_points[10], Chess_points[0]);
			double BC = calculateDistance(cross_vec_undisort[m], Chess_points[5]);
			double BD = calculateDistance(cross_vec_undisort[m], Chess_points[0]);
			double SR = (AC / BC) / (AD / BD);
			*/
			//标板坐标系的交并复比
			//棋盘格一格距离代表2mm

			/*
			double ac = (10 - 5) * Chess_Size;
			double ad = (10 - 0) * Chess_Size;
			double X1 = (ad * SR * Chess_Size * 5 - ac * Chess_Size * 0) / (ad * SR - ac);

			AC = calculateDistance(Chess_points[10], Chess_points[5]);
			AD = calculateDistance(Chess_points[10], Chess_points[4]);
			BC = calculateDistance(cross_vec_undisort[m], Chess_points[5]);
			BD = calculateDistance(cross_vec_undisort[m], Chess_points[4]);
			SR = (AC / BC) / (AD / BD);
			ac = (10 - 5) * Chess_Size;
			ad = (10 - 4) * Chess_Size;
			double X2 = (ad * SR * Chess_Size * 5 - ac * 2 * 4) / (ad * SR - ac);
			float x = (X1 + X2) / 3.0f;
			cv::Point3f Point = cv::Point3f(x, m * Chess_Size, 0);
			Che_Points.push_back(Point);
			*/
			//将标板上的交点坐标根据标板对应的外参转到相机坐标系
			/*
			cv::Mat cam_Point = (cv::Mat_<double>(3, 1) << x, m * Chess_Size, 0);
			cv::Mat Trans_Mat = (cv::Mat_<double>(3, 1) << 0, 0, 0);
			*/
			cv::Mat rotationMatrix;
			Rodrigues(Rcves[i], rotationMatrix);

			cv::Point3f Point_Plane;
			PointToCameraPoint(cross_vec_undisort[m], Point_Plane, Rcves[i], Tcves[i], IntrinxMatrix);
			vPlane_Points_Vec2.push_back(Point_Plane);
			
		}
		
		//vvPlane_Points_Vec2.push_back(vPlane_Points_Vec2);
		
		
	}
	float PlaneLight[4];
	cv::Mat PlaneLight1;
	cv::Mat PlaneLight2;
	fitPlane(vPlane_Points_Vec2, PlaneLight);
	FitLaserPlane(vPlane_Points_Vec2, PlaneLight1);
	FitPlaneRANSAC(vPlane_Points_Vec2, PlaneLight2);

	//平面拟合的误差

	std::cout << "PlaneCoe: [" << PlaneLight[0] << " ," << PlaneLight[1] << " ," << PlaneLight[2] << " ," << PlaneLight[3] << " , ] ;" << std::endl;
	Writedown(vPlane_Points_Vec2,"Cali.pcd");
	
	std::cout << "PlaneCoe1: [" << PlaneLight1.at<double>(0,0) << " ," << PlaneLight1.at<double>(1, 0) << " ," << PlaneLight1.at<double>(2, 0) << " ," << PlaneLight1.at<double>(3, 0) << " ] ;" << std::endl;
	std::cout << "PlaneCoe1: [" << PlaneLight2.at<double>(0, 0) << " ," << PlaneLight2.at<double>(0, 1) << " ," << PlaneLight2.at<double>(0, 2) << " ," << PlaneLight2.at<double>(0, 3) << " ] ;" << std::endl;
	//平面拟合的误差
	float ems = calPlaneDist(vPlane_Points_Vec2, PlaneLight[0], PlaneLight[1], PlaneLight[2], PlaneLight[3]);
	float ems1 = calPlaneDist(vPlane_Points_Vec2, PlaneLight1.at<double>(0, 0), PlaneLight1.at<double>(1, 0), PlaneLight1.at<double>(2, 0), PlaneLight1.at<double>(3, 0));
	float ems2 = calPlaneDist(vPlane_Points_Vec2, PlaneLight2.at<double>(0, 0), PlaneLight2.at<double>(0, 1), PlaneLight2.at<double>(0, 2), PlaneLight2.at<double>(0, 3));
	
	/*
	cv::Mat testimg;
	testimg = cv::imread("./1/1.jpg");

	std::vector<cv::Point2f> testLaserPoints;
	std::vector<cv::Point3f> dstPoints;
	testLaserPoints = GetLaserPoints_Steger(testimg, 150, 0, 1280, 1);
	for (int i = 0; i < testLaserPoints.size(); i++)
	{
		cv::circle(testimg, testLaserPoints[i], 1, cv::Scalar(255, 0, 0), 2, 8, 0);
	}
	ImgPtsToCamFrame(PlaneLight1, IntrinxMatrix, DistortCoeffs, testLaserPoints, dstPoints);
	Writedown(dstPoints,"test.pcd");
	*/
}

MCalibLaser::~MCalibLaser()
{
}

void MCalibLaser::LoadLaserImage(std::string& _LaserImgDir, std::vector<cv::Mat>& _CaliCamImg, std::vector <cv::Mat> &_CaliLasImg,cv::Mat &_intrinsicMatrix, cv::Mat& _coeffsMatrix)
{
	//将一组图片分成两组 前一张无激光  后一张相同但有激光
	std::vector<cv::String> ImgList;
	cv::glob(_LaserImgDir, ImgList);
	for (int i = 0; i < ImgList.size(); i++)
	{
		cv::Mat image = cv::imread(ImgList[i]);
		cv::Mat mUndisortedImg;
		//cv::undistort(image, mUndisortedImg, _intrinsicMatrix, _coeffsMatrix);
		
		if (i % 2 == 0)
			_CaliCamImg.push_back(image);
		else
			_CaliLasImg.push_back(image);

	}

}

void MCalibLaser::CalibraCamForLaserImg(std::vector<cv::Mat>& _CaliCamImg, std::vector<cv::Mat>& _Rves, std::vector<cv::Mat>& _Tves)
{
	std::vector<std::vector<cv::Point2f>> imagePoints;
	std::vector<cv::Point2f> nCorners;
	cv::Size ImgSize;
	float chesssize = 5.0f;

	std::vector<std::vector<cv::Point3f>> objectPoints;
	std::vector<cv::Point3f> realChessboardData;

	for (int i = 0; i < 8; ++i)
	{
		for (int j = 0; j < 11; ++j)
		{
			realChessboardData.push_back(cv::Point3f(j * chesssize, i * chesssize, 0.0f));
		}
	}

	for (size_t i = 0; i < _CaliCamImg.size(); ++i)
	{
		objectPoints.push_back(realChessboardData);
	}


	for (int i = 0; i < _CaliCamImg.size(); ++i)
	{
		cv::Mat image = _CaliCamImg[i];
		cvtColor(image, image, cv::COLOR_BGR2GRAY);

		//找棋盘格角点
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

	m_imagePoints = imagePoints;
	_Rves = Rvecs;
	_Tves = Tvecs;
}

std::vector<cv::Point2f> MCalibLaser::GetLaserPoints_Steger(cv::Mat& src, int gray_Thed, int Min, int Max, int Type)
{
	std::vector<cv::Point2f> LaserPoints;
	cv::Mat srcGray, srcGray1;
	cv::Mat halfimg;
	cv::Size halfsize;
	if (src.size().width > 2000)
	{
		
		halfsize = src.size() / 2;
	}
	else {
		halfsize = src.size();
	}
	cv::resize(src, halfimg, halfsize);
	cvtColor(halfimg, srcGray1, cv::COLOR_BGR2GRAY);


	//高斯滤波
	srcGray = srcGray1.clone();
	srcGray.convertTo(srcGray, CV_32FC1);
	GaussianBlur(srcGray, srcGray, cv::Size(0, 0), 4,4);


	//一阶偏导数
	cv::Mat m1, m2;
	m1 = (cv::Mat_<float>(1, 2) << 1, -1);//x方向的偏导数
	m2 = (cv::Mat_<float>(2, 1) << 1, -1);//y方向的偏导数
	cv::Mat dx, dy;
	filter2D(srcGray, dx, CV_32FC1, m1);
	filter2D(srcGray, dy, CV_32FC1, m2);
	//二阶偏导数
	cv::Mat m3, m4, m5;
	m3 = (cv::Mat_<float>(1, 3) << 1, -2, 1);   //二阶x偏导
	m4 = (cv::Mat_<float>(3, 1) << 1, -2, 1);   //二阶y偏导
	m5 = (cv::Mat_<float>(2, 2) << 1, -1, -1, 1);   //二阶xy偏导
	cv::Mat dxx, dyy, dxy;
	filter2D(srcGray, dxx, CV_32FC1, m3);
	filter2D(srcGray, dyy, CV_32FC1, m4);
	filter2D(srcGray, dxy, CV_32FC1, m5);
	//hessian矩阵
	double maxD = -1;
	std::vector<double> Pt;
	cv::Mat cv8bit;
	srcGray.convertTo(cv8bit, CV_8U);

	std::vector<cv::Point2f> points_vec;
	if (Type == 0)
	{
		Min = Min < 0 ? 0 : Min;
		Max = Max > srcGray.rows ? srcGray.rows : Max;
		for (int i = 0; i < srcGray.cols; i++)
		{
			for (int j = Min; j < Max; j++)
			{
				if (srcGray.at<uchar>(j, i) > gray_Thed)
				{
					cv::Mat hessian(2, 2, CV_32FC1);
					hessian.at<float>(0, 0) = dxx.at<float>(j, i);
					hessian.at<float>(0, 1) = dxy.at<float>(j, i);
					hessian.at<float>(1, 0) = dxy.at<float>(j, i);
					hessian.at<float>(1, 1) = dyy.at<float>(j, i);

					cv::Mat eValue;
					cv::Mat eVectors;
					eigen(hessian, eValue, eVectors);

					double nx, ny;
					double fmaxD = 0;
					if (fabs(eValue.at<float>(0, 0)) >= fabs(eValue.at<float>(1, 0)))  //求特征值最大时对应的特征向量
					{
						nx = eVectors.at<float>(0, 0);
						ny = eVectors.at<float>(0, 1);
						fmaxD = eValue.at<float>(0, 0);
					}
					else
					{
						nx = eVectors.at<float>(1, 0);
						ny = eVectors.at<float>(1, 1);
						fmaxD = eValue.at<float>(1, 0);
					}

					double t = -(nx * dx.at<float>(j, i) + ny * dy.at<float>(j, i)) / (nx * nx * dxx.at<float>(j, i) + 2 * nx * ny * dxy.at<float>(j, i) + ny * ny * dyy.at<float>(j, i));

					if (fabs(t * nx) <= 0.5 && fabs(t * ny) <= 0.5)
					{
						Pt.push_back(i);
						Pt.push_back(j);
					}
				}

			}
		}
	}
	else
	{
		Min = Min < 0 ? 0 : Min;
		Max = Max > srcGray.cols ? srcGray.cols : Max;
		for (int i = Min; i < Max; i++)
		{
			for (int j = 0; j < srcGray.rows; j++)
			{
				if (srcGray.at<float>(j, i) > gray_Thed)
				{
					cv::Mat hessian(2, 2, CV_32FC1);
					hessian.at<float>(0, 0) = dxx.at<float>(j, i);
					hessian.at<float>(0, 1) = dxy.at<float>(j, i);
					hessian.at<float>(1, 0) = dxy.at<float>(j, i);
					hessian.at<float>(1, 1) = dyy.at<float>(j, i);

					cv::Mat eValue;
					cv::Mat eVectors;
					eigen(hessian, eValue, eVectors);

					double nx, ny;
					double fmaxD = 0;
					if (fabs(eValue.at<float>(0, 0)) >= fabs(eValue.at<float>(1, 0)))  //求特征值最大时对应的特征向量
					{
						nx = eVectors.at<float>(0, 0);
						ny = eVectors.at<float>(0, 1);
						fmaxD = eValue.at<float>(0, 0);
					}
					else
					{
						nx = eVectors.at<float>(1, 0);
						ny = eVectors.at<float>(1, 1);
						fmaxD = eValue.at<float>(1, 0);
					}

					double t = -(nx * dx.at<float>(j, i) + ny * dy.at<float>(j, i)) / (nx * nx * dxx.at<float>(j, i) + 2 * nx * ny * dxy.at<float>(j, i) + ny * ny * dyy.at<float>(j, i));
					if (fabs(t * nx) <= 0.4 && fabs(t * ny) <= 0.4)
					{
						Pt.push_back(i);
						Pt.push_back(j);
					}
				}
			}
		}
	}
	for (int k = 0; k < Pt.size() / 2; k++)
	{
		cv::Point2f point;
		point.x = Pt[2 * k + 0];
		point.y = Pt[2 * k + 1];
		points_vec.push_back(point);

	}
	cv::Mat show_img;
	show_img = src.clone();

	std::vector<cv::Point2f> points_vec_merge;
	//std::vector<float> y_values;

	for (const auto& point : points_vec) 
	{
		bool found = false;
		for (int i = 0; i < points_vec_merge.size(); ++i) {
			if (abs(point.y - points_vec_merge[i].y) < 1e-5) {
				found = true;
				break;
			}
		}
		if (!found) {
			// Find all points with the same y coordinate
			std::vector<float> x_values;
			for (const auto& p : points_vec) {
				if (abs(p.y - point.y) < 1e-5) {
					x_values.push_back(p.x);
				}
			}
			// Calculate the median of x coordinates
			std::sort(x_values.begin(), x_values.end());
			float median_x = x_values[x_values.size() / 2];

			points_vec_merge.push_back(cv::Point2f(median_x, point.y));
		}
	}

	std::vector<cv::Point2f> Points_Real;
	if (src.size().width > 2000)
	{
		for (int i = 0; i < points_vec_merge.size(); i++)
		{
			cv::Point2f Real;
			Real.x = points_vec_merge[i].x * 2;
			Real.y = points_vec_merge[i].y * 2;
			Points_Real.push_back(Real);
			cv::circle(show_img, Real, 3, cv::Scalar(0, 255, 0), 1, 1, 0);
		}
	}
	else
		Points_Real = points_vec_merge;
	return Points_Real;

}

std::vector<cv::Point2f> MCalibLaser::GetLaserPoints_Sobel(const cv::Mat& inputImage)
{
	// 转换图像为灰度图
	cv::Mat grayImage;
	cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
	GaussianBlur(grayImage, grayImage, cv::Size(0, 0), 3, 3);

	// 计算 x 方向和 y 方向的 Sobel 梯度
	cv::Mat gradX, gradY;
	cv::Sobel(grayImage, gradX, CV_32F, 1, 0);
	cv::Sobel(grayImage, gradY, CV_32F, 0, 1);

	// 计算梯度强度
	cv::Mat gradientMagnitude;
	cv::magnitude(gradX, gradY, gradientMagnitude);

	std::vector<cv::Point2f> laserPoints;

	// 逐列查找最佳的轮廓点
	for (int col = 0; col < gradientMagnitude.cols; ++col) {
		cv::Mat column = gradientMagnitude.col(col);

		// 寻找最大梯度值的索引
		double maxVal;
		cv::Point maxLoc;
		cv::minMaxLoc(column, nullptr, &maxVal, nullptr, &maxLoc);

		// 将最大梯度值对应的点添加到轮廓点集合中
		laserPoints.emplace_back(static_cast<float>(maxLoc.x), static_cast<float>(maxLoc.y));
	}

	return laserPoints;
}



void MCalibLaser::GetCaliRangeAndFitLine(cv::Mat &_inputImg,int &xmin,int &ymin,int &xmax,int &ymax)
{
	cv::Mat grayinputimg;
	cv::cvtColor(_inputImg, grayinputimg, cv::COLOR_BGR2GRAY);
	cv::Size patternSize(11, 8); // 在这里假设是9x6的棋盘格

	// 用于保存检测到的角点
	std::vector<cv::Point2f> corners;

	// 寻找棋盘格角点
	bool patternFound = cv::findChessboardCorners(grayinputimg, patternSize, corners);

	if (patternFound) {
		// 如果找到棋盘格角点
		cornerSubPix(grayinputimg, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
		// 找到最左上和最右下的角点
		cv::Point2f tl = corners[0]; // 最左上角点
		cv::Point2f br = corners[patternSize.width * patternSize.height - 1]; // 最右下角点

		
		xmin = corners[0].x;
		xmax = corners[0].x;
		ymin = corners[0].y;
		ymax = corners[0].y;

		for (const auto& point : corners) 
		{
			xmin = MIN(xmin, point.x);
			xmax = MAX(xmax, point.x);
			ymin = MIN(ymin, point.y);
			ymax = MAX(ymax, point.y);
		}
	}
	else {
		std::cerr << "未能找到棋盘格角点" << std::endl;
	}
}

std::vector<cv::Point2f> MCalibLaser::FilterLaserPointsByRange(std::vector<cv::Point2f> _LaserPoints, int xmin, int ymin, int xmax, int ymax)
{
	std::vector<cv::Point2f> LaserPointsAfterFilter;
	for (int i = 0; i < _LaserPoints.size(); i++)
	{
		if ((_LaserPoints[i].x < xmax && _LaserPoints[i].x > xmin) && (_LaserPoints[i].y < ymax && _LaserPoints[i].y > ymin))
			LaserPointsAfterFilter.push_back(_LaserPoints[i]);
	}
	return LaserPointsAfterFilter;
}

std::vector<cv::Vec4f> MCalibLaser::FitChessBoardLines(const std::vector<cv::Point2f>& corners, cv::Mat& image,std::vector<std::vector<cv::Point2f>>& _Line_Points)
{
	cv::Mat showimg;
	if (image.channels() == 1)
		cv::cvtColor(image, showimg, cv::COLOR_GRAY2BGR);
	else
		showimg = image.clone();
	std::vector<cv::Vec4f> ChessBoardLines;
	int rows = 8; // 棋盘格行数
	int cols = 11; // 棋盘格列数

	for (int i = 0; i < rows; i++) 
	{
		// 提取当前行的角点
		std::vector<cv::Point2f> rowCorners;
		for (int j = 0; j < cols; j++) {
			rowCorners.push_back(corners[i * cols + j]);
		}

		// 拟合直线
		cv::Vec4f lineParams;
		cv::fitLine(rowCorners, lineParams, cv::DIST_L2, 0, 0.01, 0.01);

		// 绘制直线
		float k = lineParams[1] / lineParams[0]; // 斜率
		float b = lineParams[3] - k * lineParams[2]; // 截距
		cv::line(showimg, cv::Point(0, b-1), cv::Point(showimg.cols+1, k * showimg.cols + b), cv::Scalar(0, 255, 0), 2);

		// 存储直线参数
		ChessBoardLines.push_back(lineParams);
		_Line_Points.push_back(rowCorners);

	}
	return ChessBoardLines;
}

void MCalibLaser::PointToCameraPoint(cv::Point2f _ptImg, cv::Point3f& _ptCam, const cv::Mat _matRvecs, const cv::Mat _matTvecs, const cv::Mat _matIntrinsics)
{
	//initialize parameter
	cv::Mat rotationMatrix;//3*3
	cv::Rodrigues(_matRvecs, rotationMatrix);

	//获取图像坐标
	cv::Mat imagePoint;


	imagePoint = (cv::Mat_<double>(3, 1) << double(_ptImg.x), double(_ptImg.y), 1);
	//计算比例参数S
	double s;
	cv::Mat tempMat, tempMat2;
	tempMat = rotationMatrix.inv() * _matIntrinsics.inv() * imagePoint;
	tempMat2 = rotationMatrix.inv() * _matTvecs;
	s = tempMat2.at<double>(2, 0);
	s /= tempMat.at<double>(2, 0);

	//计算世界坐标
	cv::Mat matWorldPoints = rotationMatrix.inv() * (s * _matIntrinsics.inv() * imagePoint - _matTvecs);

	// 计算相机坐标
	cv::Mat matCameraPoints = rotationMatrix * matWorldPoints + _matTvecs;


	cv::Point3f ptCameraPoints(matCameraPoints.at<double>(0, 0), matCameraPoints.at<double>(1, 0), matCameraPoints.at<double>(2, 0));

	_ptCam = ptCameraPoints;

}

cv::Point2f MCalibLaser::getCrossPoint(const cv::Vec4f& line1, const cv::Vec4f& line2)
{
	float k1 = line1[1] / line1[0]; // 斜率
	float k2 = line2[1] / line2[0]; // 斜率
	float b1 = line1[3] - k1 * line1[2]; // 截距
	float b2 = line2[3] - k2 * line2[2]; // 截距

	// 如果两条线的角度差异很小，则它们可能平行或共线，无交点
	float angleDifference = abs(abs(k1)- abs(k2));
	if (angleDifference < 0.001 ) {
		std::cout << " parallel / coincident " << std::endl;
		return cv::Point2f(-1, -1); // 返回无效的交点
	}

	// 计算交点坐标
	float x = (b2 - b1) / (k1 - k2);
	float y = k1 * x + b1;

	return cv::Point2f(x, y);
}


void MCalibLaser::FitLaserPlane(std::vector<cv::Point3f> _vecPoints3ds, cv::Mat& _mat_plane)
{
	cv::Mat matPoints = cv::Mat(_vecPoints3ds.size(), 3, CV_64FC1);	///< 创建一个行数为数据总数，列数为3的矩阵，按顺序保存所有点

	for (int i = 0; i < _vecPoints3ds.size(); i++)
	{
		matPoints.at<double>(i, 0) = _vecPoints3ds[i].x;///< X的坐标值
		matPoints.at<double>(i, 1) = _vecPoints3ds[i].y;///< Y的坐标值
		matPoints.at<double>(i, 2) = _vecPoints3ds[i].z;///< Z的坐标值
	}

	int rows = matPoints.rows;	///< 数据点矩阵的行数
	int cols = matPoints.cols;	///< 数据点矩阵的列数

	//1. 求X、Y、Z的平均值
	cv::Mat centroid = cv::Mat::zeros(1, cols, CV_64FC1);	///< 创建一个单行、列数与数据矩阵相同的矩阵，来保存平均值
	for (int i = 0; i < cols; i++)
	{
		for (int j = 0; j < rows; j++)
		{
			centroid.at<double>(0, i) += matPoints.at<double>(j, i);	///< 累加X、Y、Z所有的元素
		}
		centroid.at<double>(0, i) /= rows;	///< 再除以数据总数，求出平均值
	}

	//2. 将点的X、Y、Z与各自平均值求差，标准差
	cv::Mat meanPoints = cv::Mat::ones(rows, cols, CV_64FC1);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			meanPoints.at<double>(i, j) = matPoints.at<double>(i, j) - centroid.at<double>(0, j);
		}
	}

	cv::Mat A, W, U, V;
	// dst = src1.inv * src2
	// 广义矩阵乘法
	cv::gemm(meanPoints, matPoints, 1, NULL, 0, A, cv::GEMM_1_T);
	//矩阵 A 将被分为U、W和V
	//A(M, N) = U(M, M) * W(M, N) * V(N, N)
	cv::SVD::compute(A, W, U, V);


	_mat_plane = cv::Mat::zeros(cols + 1, 1, CV_64FC1);	///< 4行1列的矩阵保存光平面方程的abcd
	for (int i = 0; i < cols; i++)
	{
		_mat_plane.at<double>(i, 0) = V.at<double>(cols - 1, i);
		_mat_plane.at<double>(cols, 0) += _mat_plane.at<double>(i, 0) * centroid.at<double>(0, i); ///< D = Ax + By + Cz
	}

}

//畸变矫正
void DistortPoints(std::vector<cv::Point2f> _ptSrc, std::vector<cv::Point2f>& _ptDst, const cv::Mat _matIntrinsics, const cv::Mat _matDistCoeff)
{
	if (_matIntrinsics.empty() || _matDistCoeff.empty())
	{
		return;
	}
	_ptDst.clear();

	double fx = _matIntrinsics.at<double>(0, 0);
	double fy = _matIntrinsics.at<double>(1, 1);
	double ux = _matIntrinsics.at<double>(0, 2);
	double uy = _matIntrinsics.at<double>(1, 2);

	double k1 = -_matDistCoeff.at<double>(0, 0);
	double k2 = -_matDistCoeff.at<double>(0, 1);
	double p1 = -_matDistCoeff.at<double>(0, 2);
	double p2 = -_matDistCoeff.at<double>(0, 3);
	double k3 = -_matDistCoeff.at<double>(0, 4);
	double k4 = 0;//
	double k5 = 0;//
	double k6 = 0;//

	for (int i = 0; i < _ptSrc.size(); i++)
	{
		const cv::Point2f  p = _ptSrc[i];
		//获取的点通常是图像的像素点，所以需要先通过小孔相机模型转换到归一化坐标系下；
		double xCorrected = (p.x - ux) / fx;
		double yCorrected = (p.y - uy) / fy;

		double xDistortion, yDistortion;
		//我们已知的是经过畸变矫正或理想点的坐标；
		double r2 = xCorrected * xCorrected + yCorrected * yCorrected;

		double deltaRa = 1. + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
		double deltaRb = 1 / (1. /*+ k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2*/);
		double deltaTx = 2. * p1 * xCorrected * yCorrected + p2 * (r2 + 2. * xCorrected * xCorrected);
		double deltaTy = p1 * (r2 + 2. * yCorrected * yCorrected) + 2. * p2 * xCorrected * yCorrected;
		//下面为畸变模型；
		xDistortion = xCorrected * deltaRa * deltaRb + deltaTx;
		yDistortion = yCorrected * deltaRa * deltaRb + deltaTy;
		//最后再次通过相机模型将归一化的坐标转换到像素坐标系下；
		xDistortion = xDistortion * fx + ux;
		yDistortion = yDistortion * fy + uy;

		_ptDst.push_back(cv::Point2f(xDistortion, yDistortion));
	}


}


void MCalibLaser::ImgPtsToCamFrame(const cv::Mat _LaserPlane_Coeffs, const cv::Mat _Intrinsic_Matrix, cv::Mat _Dist_Coeff,
	std::vector<cv::Point2f> _vec_ptSrc, \
	std::vector<cv::Point3f>& _vec_ptDst)
{

	_vec_ptDst.clear();

	double a = _LaserPlane_Coeffs.at<double>(0, 0), \
		   b = _LaserPlane_Coeffs.at<double>(1, 0), \
		   c = _LaserPlane_Coeffs.at<double>(2, 0), \
	       d = _LaserPlane_Coeffs.at<double>(3, 0);

	double u0 = _Intrinsic_Matrix.at<double>(0, 2), \
		   v0 = _Intrinsic_Matrix.at<double>(1, 2); //相机主点
	double fx = _Intrinsic_Matrix.at<double>(0, 0), \
		   fy = _Intrinsic_Matrix.at<double>(1, 1); //尺度因子

	std::vector<cv::Point2f> ptDistort;
	DistortPoints(_vec_ptSrc, ptDistort, _Intrinsic_Matrix, _Dist_Coeff);
	for (int i = 0; i < ptDistort.size(); i++)
	{
		

		double u = ptDistort[i].x, v = ptDistort[i].y;
		double x1 = (double)((u - u0) / fx), \
			y1 = (double)((v - v0) / fy);

		cv::Point3f pt;

		pt.x = (double)x1 * d / (a * x1 + b * y1 + c);
		pt.y = (double)y1 * d / (a * x1 + b * y1 + c);
		pt.z = (double)1 * d / (a * x1 + b * y1 + c);
		_vec_ptDst.push_back(pt);
	}
}

void MCalibLaser::DistortPoints(std::vector<cv::Point2f> _ptSrc, std::vector<cv::Point2f>& _ptDst, const cv::Mat _matIntrinsics, const cv::Mat _matDistCoeff)
{
	if (_matIntrinsics.empty() || _matDistCoeff.empty())
	{
		return;
	}
	_ptDst.clear();

	double fx = _matIntrinsics.at<double>(0, 0);
	double fy = _matIntrinsics.at<double>(1, 1);
	double ux = _matIntrinsics.at<double>(0, 2);
	double uy = _matIntrinsics.at<double>(1, 2);

	double k1 = -_matDistCoeff.at<double>(0, 0);
	double k2 = -_matDistCoeff.at<double>(1, 0);
	double p1 = -_matDistCoeff.at<double>(2, 0);
	double p2 = -_matDistCoeff.at<double>(3, 0);
	double k3 = -_matDistCoeff.at<double>(4, 0);
	double k4 = 0;//
	double k5 = 0;//
	double k6 = 0;//

	for (int i = 0; i < _ptSrc.size(); i++)
	{
		const cv::Point2f  p = _ptSrc[i];
		//获取的点通常是图像的像素点，所以需要先通过小孔相机模型转换到归一化坐标系下；
		double xCorrected = (p.x - ux) / fx;
		double yCorrected = (p.y - uy) / fy;

		double xDistortion, yDistortion;
		//我们已知的是经过畸变矫正或理想点的坐标；
		double r2 = xCorrected * xCorrected + yCorrected * yCorrected;

		double deltaRa = 1. + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
		double deltaRb = 1 / (1. /*+ k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2*/);
		double deltaTx = 2. * p1 * xCorrected * yCorrected + p2 * (r2 + 2. * xCorrected * xCorrected);
		double deltaTy = p1 * (r2 + 2. * yCorrected * yCorrected) + 2. * p2 * xCorrected * yCorrected;
		//下面为畸变模型；
		xDistortion = xCorrected * deltaRa * deltaRb + deltaTx;
		yDistortion = yCorrected * deltaRa * deltaRb + deltaTy;
		//最后再次通过相机模型将归一化的坐标转换到像素坐标系下；
		xDistortion = xDistortion * fx + ux;
		yDistortion = yDistortion * fy + uy;

		_ptDst.push_back(cv::Point2f(xDistortion, yDistortion));
	}

}

void MCalibLaser::Writedown(std::vector<cv::Point3f>& _vec_ptDst, std::string _name)
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



