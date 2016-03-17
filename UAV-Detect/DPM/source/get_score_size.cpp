#include "PM_type.h"

// get the normalized sizes for each level convolution response

vector<PM_type::INTS_2> PM_type::get_score_size(const vector<cv::Mat> &feats, const vector<cv::Mat> &filters)
{
	int num_levels = feats.size();
	int num_filters = filters.size();
	vector<INTS_2> ScoreSz( num_levels );
	for( int i=0; i<num_levels; i++ ){
		int sz[2] = { 0, 0 };
		int featRows = feats[i].rows;
		int featCols = feats[i].cols;
		for( int j=0; j<num_filters; j++ ){
			int filterRows = filters[j].rows;
			int filterCols = filters[j].cols;
			int Rows = featRows - filterRows + 1;
			int Cols = featCols - filterCols + 1;
			if( Rows>sz[0] )
				sz[0] = Rows;
			if( Cols>sz[1] )
				sz[1] = Cols;
		}
		ScoreSz[i][0] = sz[0];
		ScoreSz[i][1] = sz[1];
	}
	return ScoreSz;
}