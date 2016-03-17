#include "PM_type.h"

// add part_score (2X resolution) to root_score by subsampling

void  PM_type::sum_score(cv::Mat &rootScore, const cv::Mat &partScore, int anchorx, int anchory, int padx, int pady)
{
	int rRows = rootScore.rows, rCols = rootScore.cols;
	int pRows = partScore.rows, pCols = partScore.cols;

	cv::Mat tmp(rRows, rCols, CV_32FC1, cv::Scalar(-FLOAT_INF));

	float *rpt = (float*)tmp.data;
	float *ppt = (float*)partScore.data;
	int rstep = tmp.step1();
	int pstep = partScore.step1();

	int startx = anchorx - padx;
	int starty = anchory - pady;

	for(int ry=0; ry<rRows; ry++ ){
		int py = starty + ry*2;
		if(py<0 || py>=pRows )
			continue;
		float *pt1 = rpt + ry*rstep;
		float *pt2 = ppt + py*pstep;
		for(int rx=0; rx<rCols; rx++){
			int px = startx + rx*2;
			if(px<0 || px>=pCols)
				continue;
			pt1[rx] = pt2[px];
		}
	}

	rootScore = rootScore + tmp;
}