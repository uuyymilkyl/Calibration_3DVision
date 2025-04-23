shouban.avi shows using this demo to Reconstruct a 3D points cloud model using a simple USB camera and laser pen.

使用VS2022打开项目
打开main.py 通过注释与解注释类来执行 4个不同板块和功能的标定demo 


main.py如下：

#include "M3dRecontruct.hpp"

int main()
{
	//MCalibCam CalibCam;    //标定相机参数
	//MCalibLaser CalibLaser;  //标定光平面参数
	//MCaliHandEye CalibHand;  // 标定手眼矩阵
	M3DRecontruct ReContruct;  // 通过分割采集视频 shouban.avi 每帧提取激光像素点，来拼成重构的的3D点云图像
	//std::string videoPath = "./2.avi"; // 替换成你的视频文件路径


 }
