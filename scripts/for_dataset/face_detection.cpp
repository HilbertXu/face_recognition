//OPENCV相关头文件
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include<fstream>
#include<string>
#include<iostream>
#include<stdio.h>
#include<sstream>
#include<math.h>

//ROS相关头文件
#include<ros/ros.h>
#include<ros/package.h>
#include<std_msgs/String.h>
#include<image_transport/image_transport.h>
#include<sensor_msgs/image_encodings.h>
#include<geometry_msgs/Twist.h>
#include<cv_bridge/cv_bridge.h>

using namespace cv;
using namespace std;
int Count=0;

//订阅传感器话题以获取图像信息
image_transport::Subscriber img_sub;

//订阅语音话题获得人物的姓名
/*
 * 剧情如下：主人带来了一位客人，并对机器人说“lets welcome our new guest”
 * 识别到“new-guest”后机器人通过语音询问客人的姓名
 * 识别到后发布客人的姓名到话题/guest_name
 * 语音话题有两个一个是由语音传递开始的消息，一个是传递客人的姓名
 */
ros::Subscriber name_sub;
ros::Subscriber sp_sub;

//订阅导航节点
ros::Subscriber nav_sub;


//声明发布器
ros::Publisher gPublisher;
ros::Publisher move_pub;
ros::Publisher sp_pub; //向语音节点发布消息，提醒人们要拍照了

//TOPIC NAME
const std::string RECEIVE_IMG_TOPIC_NAME="/usb_cam/image_raw";
const std::string PUBLISH_RET_TOPIC_NAME="/ifFinish";
std_msgs::String photo_signal;
bool flag = true;//正式使用应改为false



string face_cascade_name = "/home/kamerider/catkin_ws/src/machine_vision/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
string window_name = "camera";
int count1=1;
string DataBase="/home/kamerider/catkin_ws/src/machine_vision/DataBase/";
const int IMAGE_SIZE=64;
string NAME=" ";

string int2str( int val )
{
    ostringstream out;
    out<<val;
    return out.str();
}

int max(int a,int b)
{
	if(a>b)
	{
		return a;
	}
	else
		return b;
}

Mat resize_image(Mat img)
{
	int top=0;
	int bottom=0;
	int left=0;
	int right=0;

	int width = img.rows;
	int height = img.cols;

	int longest_edge = max(width,height);

	if(height< longest_edge)
	{
		int dh = longest_edge-height;
		int top=int(dh/2);
	    int bottom = dh -top;
	}
	else
	{
		int dw = longest_edge - width;
		int left = int(dw/2);
		int right = dw - left;
	}
	//给图像增加边界
	Mat dst;
	copyMakeBorder(img ,dst,top,bottom,left,right,BORDER_CONSTANT,Scalar(0,0,0));
	Mat result;
	cv::resize(dst,result,Size(IMAGE_SIZE,IMAGE_SIZE));
	return result;
}

void detectAndDisplay(Mat frame)
{
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
	if(faces.size()!=0 && count1<=200)
	{
		Mat cut_img;
		string savePATH=DataBase + NAME + "/" + int2str(count1) + ".jpg";
		cout<<savePATH<<" saved "<<"  ";
		for(int i=0;i<faces.size();i++)
		{
			Point rec1(faces[i].x,faces[i].y);
			Point rec2(faces[i].x+faces[i].width,faces[i].y+faces[i].height);
			cut_img = frame(Range(faces[i].y-50,faces[i].y + faces[i].height+50), Range(faces[i].x-50,faces[i].x + faces[i].width+50));
		}
		imshow("original_img",frame);
		Mat res=resize_image(cut_img);
		imshow("resize_img",res);
		imwrite(savePATH,res);
		count1++;
	}
    for( int i = 0; i < faces.size(); i++ )
	{
		cv::rectangle(frame,faces[i],Scalar(0,0,255),1);
		cout<<faces[i]<<endl;
	}
}
void nameCallback(const std_msgs::String::ConstPtr& msg)
{
	NAME = msg->data; //暂时默认来客姓名是JACK
}

void speechCallback(const std_msgs::String::ConstPtr& msg)
{
	if(msg->data == "welcome") //主人说一句"lets welcome our guest"识别其中的welcome
	{
		flag = true;
		//语音节点说"OK i will take photo of new guset and remember him"
	}
}

void imgCallback(const sensor_msgs::ImageConstPtr& msg)
{
	if(flag == true && NAME != " ")
	{
		ROS_INFO("IMAGE RECEIVED");
		std_msgs::String signal;
		std::stringstream ss;
		ss<<"finish";
		signal.data = ss.str();

		if(Count==200)
		{
			cout<<"100 photos saved\n";
			img_sub.shutdown();
			gPublisher.publish(signal);
			sp_pub.publish(signal);
		}

		//使用cv_bridge将传感器信息转换成opencv中的Mat型对象
		cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
		cv::Mat img = cv_ptr->image;
		if(face_cascade.load(face_cascade_name));
		{
			ROS_INFO("[OPENCV ERROR] Cannot load CLASSIFIER!\n");
		}
		detectAndDisplay(img);
		imshow(window_name,img);
		Count++;
		waitKey(30);
	}
}
int main(int argc, char **argv)

{
	//以下是ros部分
	ros::init(argc, argv, "face_detection");
	ROS_INFO("----------INIT----------");

	ros::NodeHandle nh;
	move_pub = nh.advertise<geometry_msgs::Twist>("cmd_vel_mux/input/navi",1); //移动
	sp_pub = nh.advertise<std_msgs::String>("img2speech",1); //拍照片的信号
    name_sub = nh.subscribe("guest_name",1,nameCallback);
	//sp_sub = nh.subscribe("speech2img",1,speechCallback);

	gPublisher = nh.advertise<std_msgs::String>(PUBLISH_RET_TOPIC_NAME,1);

	//IMAGE TRANSPORT
	image_transport::ImageTransport it(nh);
	img_sub = it.subscribe(RECEIVE_IMG_TOPIC_NAME,1 , imgCallback);
	ros::spin();
	return 0;
}
