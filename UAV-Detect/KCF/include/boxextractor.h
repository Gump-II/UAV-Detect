#ifndef BOXEXTRACTOR_H
#define BOXEXTRACTOR_H
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;


class BoxExtractor {
public:
	BoxExtractor();
	Rect extract(Mat img);
	Rect extract(const std::string& windowName, Mat img);

	struct handlerT{
		bool isDrawing;
		Rect box;
		Mat image;

		// initializer list
		handlerT() : isDrawing(false) {};
	}params;

private:
	static void mouseHandler(int event, int x, int y, int flags, void *param);
	void opencv_mouse_callback(int event, int x, int y, int, void *param);
};

#endif // BOXEXTRACTOR_H
