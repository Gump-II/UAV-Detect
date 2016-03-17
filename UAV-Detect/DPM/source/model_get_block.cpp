#include "PM_type.h"

void PM_type::model_get_block(MODEL &model, MODEL::FILTERS &obj, cv::Mat &f)
{
	int	bl = obj.blocklabel;
	char type = model.blocks[bl].type;
	int	*shape = model.blocks[bl].shape;
	vector<float> &w = model.blocks[bl].w;

	int dim = shape[0] * shape[1] * shape[2];
	if( dim!=w.size() )
		throw	runtime_error("");

	cv::Mat	f0(shape[0], shape[1], CV_32FC(shape[2]), &w[0]);
	f.create( shape[0], shape[1], CV_32FC(shape[2]) );
	f0.copyTo(f);

	// Flip (if needed) according to block type
	switch(type){
		case block_types_Filter:
			if(obj.flip )
				flipfeat( f, 0 );			
			break;
		case block_types_PCAFilter:
			if(obj.flip )
				flipfeat( f, 1 );
			break;
		case block_types_SepQuadDef:
			if(obj.flip ){
				int	r,c,d;
				yuInd2Sub( 1, shape, r, c, d );
				float	*pt = f.ptr<float>(r,c);
				pt[d] = -pt[d];
			}
			break;
		default:
			break;
	}

	return;
}

void PM_type::model_get_block(MODEL &model, int bl, cv::Mat &f)
{
	char type = model.blocks[bl].type;
	int	*shape = model.blocks[bl].shape;
	vector<float> &w = model.blocks[bl].w;

	int	dim = shape[0] * shape[1] * shape[2];
	if( dim!=w.size() )
		throw runtime_error("");

	cv::Mat	f0(shape[0], shape[1], CV_32FC(shape[2]), &w[0]);
	f.create( shape[0], shape[1], CV_32FC(shape[2]) );
	f0.copyTo(f);

	return;
}

cv::Mat	PM_type::model_get_block(MODEL &model, int bl, bool flip)
{
	char type = model.blocks[bl].type;
	int	*shape = model.blocks[bl].shape;
	vector<float> &w = model.blocks[bl].w;

	int	dim = shape[0] * shape[1] * shape[2];
	if( dim!=w.size() )
		throw	runtime_error("");

	cv::Mat	f0(shape[0], shape[1], CV_32FC(shape[2]), &w[0]);
	cv::Mat	f(shape[0], shape[1], CV_32FC(shape[2]));
	f0.copyTo(f);

	// Flip (if needed) according to block type
	switch(type){
		case block_types_Filter:
			if(flip )
				flipfeat( f, 0 );			
			break;
		case block_types_PCAFilter:
			if(flip )
				flipfeat( f, 1 );
			break;
		case block_types_SepQuadDef:
			if(flip ){
				int		r,c,d;
				yuInd2Sub( 1, shape, r, c, d );
				float	*pt = f.ptr<float>(r,c);
				pt[d] = -pt[d];
			}
			break;
		default:
			break;
	}

	return	f;
}