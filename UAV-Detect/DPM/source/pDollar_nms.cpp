#include "PM_type.h"

cv::Mat PM_type::pDollar_nms(const cv::Mat &bbs, float overlap)
{
	// extract some rows of bbs, assume the first 4 columns are [x1,y1,x2,y2] of the detected bounding box
	int n = bbs.rows;

	// make sure that all bbs have been sorted, that is : high score bbs appear in top rows, low score bbs appear in bottom rows
	vector<int>	Ind( n, 0 );
	cv::Mat	Idx(n, 1, CV_32SC1, &Ind[0]);
	const cv::Mat &score = bbs.col(bbs.cols - 1);
	sortIdx( score, Idx, CV_SORT_EVERY_COLUMN+CV_SORT_DESCENDING );

	// for each i suppress all j st j>i and area-overlap>overlap
	vector<bool> kp( n, true );
	const cv::Mat &xs = bbs.col(0); // xstart
	const cv::Mat &ys = bbs.col(1); // ystart
	const cv::Mat &xe = bbs.col(2); // xend
	const cv::Mat &ye = bbs.col(3); // yend
	
	cv::Mat w = xe - xs;
	cv::Mat h = ye - ys;
	cv::Mat as = w.mul(h);
	for(int ii=0; ii<n; ii++ ){		
		int i = Ind[ii];
		for(int jj = ii+1; jj<n; jj++ ){
			int j = Ind[jj];
			if(!kp[j] )
				continue;
			//float iw = MIN(xe[i],xe[j]) -  MAX(xs[i],xs[j]);
			float iw = MIN(xe.at<float>(i),xe.at<float>(j)) - MAX(xs.at<float>(i),xs.at<float>(j));
			if(iw<0 )
				continue;
			//float ih = MIN(ye[i],ye[j]) - MAX(ys[i],ys[j]);
			float ih = MIN(ye.at<float>(i),ye.at<float>(j)) - MAX(ys.at<float>(i),ys.at<float>(j));
			if(ih<0 )
				continue;
			float o = iw * ih;
			//float u = MIN(as[i],as[j]);
			float u = MIN(as.at<float>(i),as.at<float>(j));
			float ratio = o / u;
			if(ratio>overlap )
				kp[j] = false;
		}
	}

	int cnt = 0;
	for(int i=0; i<n; i++ )
		if(kp[i])
			cnt++;
	cv::Mat ret( cnt, bbs.cols, CV_32FC1 );
	cnt = 0;
	for(int i=0; i<n; i++ )
		if(kp[i])
			bbs.row(i).copyTo( ret.row(cnt++) );
	return ret;
}