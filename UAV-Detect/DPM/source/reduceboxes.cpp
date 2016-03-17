#include "PM_type.h"

namespace PM_type{
	void reduceboxes(const MODEL &model, cv::Mat &bs)
		//% Eliminate columns for filters that are not used.
		//%   b = reduceboxes(model, bs)
		//%
		//%   E.g., [0 0 0 0 10 20 110 120] -> [10 20 110 120]
		//%   Index end-1 is the component label and index end is the 
		//%   detection score.
		//%
		//%   This function assumes that model is a mixture model where
		//%   each component always places exactly the same number of filters.
		//%
		//% Return value
		//%   b       Filter bounding boxes with unused filter columns removed
		//% Arguments
		//%   model   Object model
		//%   bs      Filter bounding boxes
	{
		// Only reduce boxes for mixtures of star models
		if(model.type!=model_types_MixStar )
			return;

		int	n = model.rules[model.start][0].rhs.size();
		// n*4+2 := 4 coordinates per boxes plus the component index and score as well as level-index
		cv::Mat	b(bs.rows, n * 4 + 3, bs.type(), cv::Scalar(0));

		cv::Mat	bs_col = bs.col(bs.cols - 3);
		double	maxc = 0;
		minMaxLoc( bs_col, NULL, &maxc );
		cv::Mat	bs_int;
		bs_col.convertTo( bs_int, CV_32SC1 );

		for(int idx=1; idx<=(int)maxc; idx++ ){
			vector<int>		Ind;
			for(int i=0; i<bs.rows; i++ )
				if(bs_int.at<int>(i)==idx )
					Ind.push_back(i);
			if(Ind.empty())
				continue;

			cv::Mat	tmp(Ind.size(), bs.cols, bs.type());
			for(unsigned i=0; i<Ind.size(); i++ )
				bs.row( Ind[i] ).copyTo( tmp.row(i) );

			vector<int>	resv;
			for(int j=0; j<bs.cols-3; j+=4 ){
				cv::Mat	tmp2 = tmp.colRange(j, j + 4);
				int	cnt = countNonZero( tmp2 );
				if(cnt>0 ){
					resv.push_back(j);
					resv.push_back(j+1);
					resv.push_back(j+2);
					resv.push_back(j+3);
				}
			}
			resv.push_back( bs.cols-3 );
			resv.push_back( bs.cols-2 );
			resv.push_back( bs.cols-1 );
			if(resv.size()!=b.cols )
				throw runtime_error("");

			cv::Mat	tmp_resv(Ind.size(), b.cols, b.type());
			for( int i=0; i!=resv.size(); i++ )
				tmp.col( resv[i] ).copyTo( tmp_resv.col(i) );

			for( int i=0; i!=Ind.size(); i++ )
				tmp_resv.row(i).copyTo( b.row(Ind[i]) );
		}

		bs.create( b.rows, b.cols, bs.type() );
		b.copyTo(bs);
	}
}