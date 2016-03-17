#include "PM_type.h"
using namespace PM_type;

inline void max_filter_1d_v2(const float *vals, float *out_vals, int *I, int s, int step, int n, FLOATS_SSS &DEFCACHE) 
//	input:
//		vals[] : whose length is "n", step-size of its element is "step"
//		s : how many neighborhoods are involved in calculating the deformation cost
//		a,b : the coefficients in cost calculation
//	output:
//		out_vals[] & I[] : both has the same size as vals[]
//			out_vals[] stores the max deformation costs
//			I[] stores the neighborhood's position to which the deformation cost achieves max
//
// max_filter_1d() will calculate the deformation cost of each vals[i] to its 4 neighbors, namely
//			float val = *(vals + j*step) - a*square(i-j) - b*(i-j);
// where a&b are coeffs
// *(vals + j*step) is one of the 4 neighbors of vals[i] 
{
	float	*outpt = out_vals;
	int		*Ipt = I;
	for (int i = 0; i < n; i++) {
		float max_val = (float)(-1e20);
		int argmax     = 0;
		int first = MAX(0, i-s);
		int last = MIN(n-1, i+s);
		const float	*vals_off = vals + first*step;
		for (int j = first; j <= last; j++) {
			float val = *vals_off + DEFCACHE[i-j+s];
			if (val > max_val) {
				max_val = val;
				argmax  = j;
			}
			vals_off += step;
		}
		*outpt = max_val;
		*Ipt = argmax;
		outpt += step;
		Ipt += step;
	}
}

// matlab entry point
// [M, Ix, Iy] = bounded_dt(vals, ax, bx, ay, by, s)
void	PM_type::bounded_dt2(const cv::Mat &vals, FLOATS_SSS &DXDEF, FLOATS_SSS &DYDEF, cv::Mat &M, cv::Mat &Ix, cv::Mat &Iy)
{
	const int s = PM_type::SSS;

	// storage allocation
	int	Rows = vals.rows, Cols = vals.cols;
	if( !Rows || !Cols )
		throw runtime_error("");
	M.create( Rows, Cols, CV_32FC1 );
	Ix.create( Rows, Cols, CV_32SC1 );
	Iy.create( Rows, Cols, CV_32SC1 );
	cv::Mat	tmpM(Rows, Cols, CV_32FC1);
	cv::Mat	tmpIy(Rows, Cols, CV_32SC1);

	const float	*vPt = (float*)vals.data; int vStep = vals.step1();	
	float	*mPt = (float*)M.data; int mStep = M.step1();
	int		*ixPt = (int*)Ix.data; int ixStep = Ix.step1();
	int		*iyPt = (int*)Iy.data; int iyStep = Iy.step1();
	float	*tmPt = (float*)tmpM.data; int tmStep = tmpM.step1();
	int		*tyPt = (int*)tmpIy.data; int tyStep = tmpIy.step1();

	for( int c=0; c<Cols; c++ )
		max_filter_1d_v2( vPt++, tmPt++, tyPt++, s, Cols, Rows, DYDEF );

	tmPt = (float*)tmpM.data;
	for( int r=0; r<Rows; r++ ){
		max_filter_1d_v2( tmPt, mPt, ixPt, s, 1, Cols, DXDEF );
		tmPt += tmStep;
		mPt += mStep;
		ixPt += ixStep;
	}

	ixPt = (int*)Ix.data;
	iyPt = (int*)Iy.data;
	tyPt = (int*)tmpIy.data;
	for( int r=0; r<Rows; r++ ){
		for( int c=0; c<Cols; c++ )
			*(iyPt++) = tyPt[*(ixPt++)];
		tyPt += tyStep;
	}

}
