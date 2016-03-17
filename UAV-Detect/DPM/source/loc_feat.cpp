#include "PM_type.h"

namespace PM_type{
	void	loc_feat(const MODEL &model, int num_levels, cv::Mat &f)
		//% Location and scale features.
		//%   f = loc_feat(model, num_levels)
		//%
		//%   Compute a feature indicating if level i is in the first octave of 
		//%   the feature pyramid, the second octave, or above. This could be
		//%   generalized to a "location feature" instead of just a scale feature.
		//%
		//% Return value
		//%   f = [f_1, ..., f_i, ..., f_num_levels],
		//%   where f_i is the 3x1 vector = 
		//%     [1; 0; 0] if 1 <= i <= model.interval
		//%     [0; 1; 0] if model.interval+1 <= i <= 2*model.interval
		//%     [0; 0; 1] if 2*model.interval+1 <= i <= num_levels
		//%
		//% Arguments
		//%   model       Model used for detection
		//%   num_levels  Number of levels in the feature pyramid
	{
		f.create( 3, num_levels, CV_32FC1 );
		f = cv::Scalar(0);

		int		b = 0;
		int		e = MIN( num_levels, model.interval );
		f(cv::Rect(b, 0, e - b, 1)) = cv::Scalar(1);

		b = e;
		e = MIN( num_levels, 2*e );
		f(cv::Rect(b, 1, e - b, 1)) = cv::Scalar(1);

		b = e;
		f(cv::Rect(b, 2, f.cols - b, 1)) = cv::Scalar(1);

		return;
	}
}