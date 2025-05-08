----------------------------------
使用线激光发射器+USB工业相机+机械臂 进行机械臂平扫点云生成算法验证，流程如下
1.标定相机参数
2.标定光平面参数
3.标定手眼矩阵
4.相机拍摄平扫视频，同时机械臂通过手眼矩阵记录相机位姿信息
5.通过坐标系变换方法得到每帧点云，逐帧拼接成完整点云图像
-----------------------------------

shouban.avi shows using this demo to Reconstruct a 3D points cloud model using a simple USB camera and laser pen.
-----------------------------------

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
