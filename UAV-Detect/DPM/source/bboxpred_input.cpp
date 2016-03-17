#include "PM_type.h"

void	PM_type::bboxpred_input(const cv::Mat &ds, const cv::Mat &bs, cv::Mat &A, cv::Mat &x1, cv::Mat &y1, cv::Mat &x2, cv::Mat &y2, cv::Mat &w, cv::Mat &h)
//
//% Construct input for the bbox predictor from detections
//% and filter bounding boxes.
//%   [A, x1, y1, x2, y2, w, h] = bboxpred_input(ds, bs)
//%
//%   If beta_x1 is a vector of vector of learned regression coefficients for
//%   predicting the new location of the x1 component of a bounding box,
//%   the new x1 is predicted as:					   
//%
//%     dx1 = A*beta_x1;
//%     x1 = x1 + w*dx1;
//%
//%   Computing x2, y1, and y2 are similar.
//%
//% Return values
//%   A       Each row is a feature vector (predictor variables)
//%   x1      Original detection window coordinates
//%   y1        ...
//%   x2        ...
//%   y2        ...
//%   w       Widths of original detection windows
//%   h       Heights of original detection windows
//%
//% Arguments
//%   ds     Detection window coordinates
//%   bs     Coordinates for each filter placed in the detection
{
	// detection windows' coordinates
	x1 = ds.col(0);
	y1 = ds.col(1);
	x2 = ds.col(2);
	y2 = ds.col(3);
	// detection windows' widths and heights
	w = x2 - x1;
	h = y2 - y1;
	// detection windows' centers
	cv::Mat	rx = x1 + w*0.5f;
	cv::Mat	ry = y1 + h*0.5f;

	int		A_cols = (int)(bs.cols/4.f) * 2 + 1;
	A.create(bs.rows,A_cols,CV_32FC1);
	int		i = 0;
	for( int j=3; j<bs.cols; j+=4 ){
		// filters' centers
		cv::Mat px = bs.col(j - 3) + (bs.col(j - 1) - bs.col(j - 3))*0.5;
		cv::Mat py = bs.col(j - 2) + (bs.col(j) - bs.col(j - 2))*0.5;
		A.col(i++) = ( px - rx ) / w;
		A.col(i++) = ( py - ry ) / h;
	}
	if( i!=A_cols-1 )
		throw runtime_error("");
	A.col(i) = cv::Scalar(1);
}
