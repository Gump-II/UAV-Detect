#include "PM_type.h"

void	PM_type::bboxpred_get(const MODEL &model, cv::Mat &ds, const cv::Mat &bs)
//% Get predicted bounding boxes.
//%   [bbox, bs_out] = bboxpred_get(bboxpred, ds, bs)
//%
//% Return values
//%   ds_pred   Output detection windows
//%   bs_pred   Output filter bounding boxes
//%
//% Arguments
//%   bboxpred  Bounding box prediction coefficients (see bboxpred_train.m)
//%   ds        Source detection windows
//%   bs        Source filter bounding boxes
{
	if( model.bboxpred.empty() )
		return;

	cv::Mat comp1 = ds.col(ds.cols - 2);
	int ncomps = model.RuleData.size();
	cv::Mat comp; comp1.convertTo(comp, CV_32SC1);
	const int * const compPt = (int*)comp.data;

	cv::Mat	ds_pred(ds.rows, ds.cols, ds.type());
	int		pred_id = 0;

	for( int c=0; c<ncomps; c++ ){
		// limit boxes to just component c
		vector<int>		cinds;
		const int *pt = compPt;
		for( int i=0; i<ds.rows; i++ )
			if( *(pt++)==c )
				cinds.push_back(i);
		// if no detections of component c
		if( cinds.empty() )
			continue;
		// extract those detections of component c
		cv::Mat	b(cinds.size(), bs.cols, bs.type());
		for( unsigned i=0; i<cinds.size(); i++ )
			bs.row( cinds[i] ).copyTo( b.row(i) );
		cv::Mat	d(cinds.size(), ds.cols, ds.type());
		for( unsigned i=0; i<cinds.size(); i++ )
			ds.row( cinds[i] ).copyTo( d.row(i) );
		// build test data
		cv::Mat	A;
		cv::Mat	x1, y1, x2, y2, w, h;
		bboxpred_input( d, b, A, x1, y1, x2, y2, w, h );
		// predict displacements
		cv::Mat	dx1 = A * model.bboxpred[c].x1;
		cv::Mat	dy1 = A * model.bboxpred[c].y1;;
		cv::Mat	dx2 = A * model.bboxpred[c].x2;
		cv::Mat	dy2 = A * model.bboxpred[c].y2;
		// compute object location from predicted displacements
		cv::Mat	ds_pred_sub = ds_pred.rowRange(pred_id, pred_id + cinds.size());
		pred_id += cinds.size();
		if( dx1.rows!=ds_pred_sub.rows )
			throw	runtime_error("");
		ds_pred_sub.col( 0 ) = x1 + w.mul(dx1);
		ds_pred_sub.col( 1 ) = y1 + h.mul(dy1);
		ds_pred_sub.col( 2 ) = x2 + w.mul(dx2);
		ds_pred_sub.col( 3 ) = y2 + h.mul(dy2);
		d.colRange(4,ds.cols).copyTo( ds_pred_sub.colRange(4,ds.cols) );
	}

	ds.create( ds_pred.rows, ds_pred.cols, ds_pred.type() );
	ds_pred.copyTo(ds);

}
