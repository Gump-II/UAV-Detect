#include "PM_type.h"

cv::Mat	PM_type::nms(const cv::Mat &boxes, float overlap)
//% Non-maximum suppression.
//%   pick = nms(boxes, overlap) 
//% 
//%   Greedily select high-scoring detections and skip detections that are 
//%   significantly covered by a previously selected detection.
//%
//% Return value
//%   pick      Indices of locally maximal detections
//%
//% Arguments
//%   boxes     Detection bounding boxes (see pascal_test.m)
//%   overlap   Overlap threshold for suppression
//%             For a selected box Bi, all boxes Bj that are covered by 
//%             more than overlap are suppressed. Note that 'covered' is
//%             is |Bi \cap Bj| / |Bj|, not the PASCAL intersection over 
//%             union measure.
{
	if( boxes.empty() )
		return boxes;

	cv::Mat	x1 = boxes.col(0);
	cv::Mat	y1 = boxes.col(1);
	cv::Mat	x2 = boxes.col(2);
	cv::Mat	y2 = boxes.col(3);
	cv::Mat	s = boxes.col(boxes.cols - 1);

	cv::Mat	area = x2 - x1 + 1;
	area = area.mul(y2-y1+1);

	vector<int>	Ind( s.rows, 0 );
	cv::Mat	Idx(s.rows, 1, CV_32SC1, &Ind[0]);
	sortIdx( s, Idx, CV_SORT_EVERY_COLUMN+CV_SORT_ASCENDING );

	vector<int>	pick;
	while( !Ind.empty() ){
		int	last = Ind.size() - 1;
		int	i = Ind[last];
		pick.push_back(i);

		vector<int>	suppress( 1, last );
		for( int pos=0; pos<last; pos++ ){
			int		j = Ind[pos];
			float	xx1 = std::max(x1.at<float>(i), x1.at<float>(j));
			float	yy1 = std::max(y1.at<float>(i), y1.at<float>(j));
			float	xx2 = std::min(x2.at<float>(i), x2.at<float>(j));
			float	yy2 = std::min(y2.at<float>(i), y2.at<float>(j));
			float	w = xx2-xx1+1;
			float	h = yy2-yy1+1;
			if( w>0 && h>0 ){
				// compute overlap 
				float	area_intersection = w * h;
				float	o1 = area_intersection / area.at<float>(j);
				float	o2 = area_intersection / area.at<float>(i);
				float	o = std::max(o1,o2);
				if( o>overlap )
					suppress.push_back(pos);
			}
		}

		std::set<int>	supp( suppress.begin(), suppress.end() );
		vector<int>		Ind2;
		for( int i=0; i!=Ind.size(); i++ ){
			if( supp.find(i)==supp.end() )
				Ind2.push_back( Ind[i] );
		}
		Ind = Ind2;

	}

	cv::Mat	ret(pick.size(), boxes.cols, boxes.type());
	for( unsigned i=0; i<pick.size(); i++ )
		boxes.row( pick[i] ).copyTo( ret.row(i) );
	
	return ret;
}