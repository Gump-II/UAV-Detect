#include "boxextractor.h"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
BoxExtractor::BoxExtractor()
{
}

void BoxExtractor::mouseHandler(int event, int x, int y, int flags, void *param){
	BoxExtractor *self = static_cast<BoxExtractor*>(param);
	self->opencv_mouse_callback(event, x, y, flags, param);
}

void BoxExtractor::opencv_mouse_callback(int event, int x, int y, int, void *param){
	handlerT * data = (handlerT*)param;
	switch (event){
		// update the selected bounding box
	case EVENT_MOUSEMOVE:
		if (data->isDrawing){
			data->box.width = x - data->box.x;
			data->box.height = y - data->box.y;
		}
		break;

		// start to select the bounding box
	case EVENT_LBUTTONDOWN:
		data->isDrawing = true;
		data->box = cvRect(x, y, 0, 0);
		break;

		// cleaning up the selected bounding box
	case EVENT_LBUTTONUP:
		data->isDrawing = false;
		if (data->box.width < 0){
			data->box.x += data->box.width;
			data->box.width *= -1;
		}
		if (data->box.height < 0){
			data->box.y += data->box.height;
			data->box.height *= -1;
		}
		break;
	}
}

Rect BoxExtractor::extract(Mat img){
	return extract("Bounding Box Extractor", img);
}

Rect BoxExtractor::extract(const std::string& windowName, Mat img){

	int key = 0;

	// show the image and give feedback to user
	//imshow(windowName, img);
	//printf("Select an object to track and then press SPACE/BACKSPACE/ENTER button!\n");

	// copy the data, rectangle should be drawn in the fresh image
	params.image = img.clone();

	// select the object
	setMouseCallback(windowName, mouseHandler, (void *)&params);


	// draw the selected object
	rectangle(
		params.image,
		params.box,
		Scalar(255, 0, 0), 2, 1
		);

	// show the image bouding box
	imshow(windowName, params.image);

	// reset the image
	params.image = img.clone();

	//get keyboard event



	return params.box;
}

