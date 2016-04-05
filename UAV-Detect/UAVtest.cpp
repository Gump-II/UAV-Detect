#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "kcftracker.hpp"
#include "dirent.h"
#include "boxextractor.h"

#include "FastDPM.h"
#include <conio.h>

using namespace std;


int main(int argc, char* argv[]){


	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool LAB = false;
	bool ENTER_LEAVE = true;
	bool FRAMES_FIRST = true;
	int nFrames = 0;// Frame counter

	string model_path("UAV_2.txt");//DPM model

	cv::Mat frame;
	cv::Mat image;
	cv::VideoCapture capture("uav_data/6.avi");
	//cv::VideoCapture capture(0);
	if (model_path.empty()){
		cout << "Please input model file...." << endl;
		return -1;
	}

	if (!capture.isOpened()){
		cout << "Fail to open video....." << endl;
		return -1;
	}

	//init KCF
	BoxExtractor extractor;
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);// Create KCFTracker object
	Rect result;// Tracker results
	ofstream resultsFile;
	string resultsPath = "output.txt";	// Write Results
	resultsFile.open(resultsPath);


	FastDPM	PM(model_path);

	while (true){

		capture >> frame;
		if (frame.empty()){
			cout << "Fail to get video......." << endl;
			break;
		}

		if (ENTER_LEAVE)
		{
			//DPM detect UAV

			frame.copyTo(image);
			cv::Mat	img = PM.prepareImg(image);
			PM.detect(img, -1.0f, true, true);
		
			if (!PM.detections.empty())
			{
				ENTER_LEAVE = false;
				FRAMES_FIRST = true;
			}
		}

	

		else{

			//KCF track UAV
	
			double t = (double)cvGetTickCount();
			// First frame, give the groundtruth to the tracker
			if (FRAMES_FIRST) {
				//relate KCF with DPM
				cv::Rect box(int(UL_INIT.x), int(UL_INIT.y), int(BR_INIT.x - UL_INIT.x), int(BR_INIT.y - UL_INIT.y));//box(x,y,width,height)
				tracker.init(box, frame);
				rectangle(frame, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(0, 255, 255), 1, 8);
				FRAMES_FIRST = false;
				//rectangle(frame, Point(400,400), Point(638,478), Scalar(0, 255, 255), 1, 8);

			}
			// Update
			else{
				result = tracker.update(frame);
				rectangle(frame, Point(result.x, result.y), Point(result.x + result.width, result.y + result.height), Scalar(0, 255, 255), 1, 8);
				resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << endl;
				//DPM model detect UAV's leave
				if (!(result.x > 0 && result.y > 0 && (640 - result.x - result.width) > 0 && (480 - result.y - result.height) > 0)){
					ENTER_LEAVE = true;
					cout << "UAV has left....." << endl;
				}
			}

			t = (double)cvGetTickCount() - t;
			cout << "COST TIME: " << t / ((double)cvGetTickFrequency()*1000.) << endl;
		
		}


		imshow("UAV DETECT", frame);
		char key = waitKey(1);
		if (key == 27) break;
		
	}

	resultsFile.close();							
	cvDestroyWindow("UAV DETET");
	return 0;

}