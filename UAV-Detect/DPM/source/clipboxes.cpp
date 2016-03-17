#include "PM_type.h"

// Clip detection windows to image the boundary.
void  PM_type::clipboxes(const int imsize[2], cv::Mat &ds, cv::Mat &bs)
{
	if( !ds.empty() ){
		cv::Mat	ds_x1 = ds.col(0);
		cv::Mat	ds_y1 = ds.col(1);
		cv::Mat	ds_x2 = ds.col(2);
		cv::Mat	ds_y2 = ds.col(3);
		ds_x1 = cv::max( ds_x1, 0 );
		ds_y1 = cv::max( ds_y1, 0 );
		ds_x2 = cv::min( ds_x2, imsize[1]-1 );
		ds_y2 = cv::min( ds_y2, imsize[0]-1 );
		// remove invalid detections
		cv::Mat	w = ds_x2 - ds_x1;
		cv::Mat	h = ds_y2 - ds_y1;
		cv::Mat	invalids = ((w<1) | (h<1));
		// 
		int		cnt = countNonZero( invalids );
		cv::Mat	ds_cpy(ds.rows - cnt, ds.cols, ds.type());
		int		j = 0;
		for( int i=0; i<ds.rows; i++ ){
			if( invalids.at<uchar>(i) )
				continue;
			ds.row(i).copyTo( ds_cpy.row(j) );
			j++;
		}
		ds.create( ds_cpy.rows, ds_cpy.cols, ds_cpy.type() );
		ds_cpy.copyTo(ds);
		//
		if( !bs.empty() ){
			cv::Mat	bs_cpy(bs.rows - cnt, bs.cols, bs.type());
			int		j = 0;
			for( int i=0; i<bs.rows; i++ ){
				if( invalids.at<uchar>(i) )
					continue;
				bs.row(i).copyTo( bs_cpy.row(j) );
				j++;
			}
			bs.create( bs_cpy.rows, bs_cpy.cols, bs_cpy.type() );
			bs_cpy.copyTo(bs);
		}
	}
}