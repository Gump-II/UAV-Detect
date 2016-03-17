#include "PM_type.h"

void	fconv_1(const cv::Mat &A, const cv::Mat &F, cv::Mat &R);
void	fconv_SSE(const cv::Mat &A, const cv::Mat &F, cv::Mat &R);

void PM_type::fconv(const vector<cv::Mat> &Feats, const cv::Mat &Filter, vector<cv::Mat> &Response, int lev1, int lev2, vector<INTS_2> &ScoreSz)
// The filter response of Filter on HOG feature map Feats
{
	Response.resize( Feats.size() );
	for( int i=lev1; i<lev2; i++ ){
		Response[i].create( ScoreSz[i][0], ScoreSz[i][1], CV_32FC1 );
		Response[i] = cv::Scalar(-FLOAT_INF);
		fconv_SSE( Feats[i], Filter, Response[i] );
	}
}

void	fconv_1(const cv::Mat &A, const cv::Mat &F, cv::Mat &R)
{
	int		RowA = A.rows, ColA = A.cols, NumFeatures = A.channels();
	int		RowF = F.rows, ColF = F.cols, ChnF = F.channels();
	if( NumFeatures!=ChnF )
		throw runtime_error("");

	int    RowR = RowA - RowF + 1, ColR = ColA - ColF + 1;

	float *Rpt = (float*)R.data;
	int Rstep = R.step1();

	for( int r=0; r!=RowR; r++ ){
		float *pt = Rpt + r*Rstep;
		for( int c=0; c!=ColR; c++ ){
			cv::Mat	Asub = A(cv::Rect(c, r, ColF, RowF));
			*(pt++) = (float)( F.dot( Asub ) );
		}
	}
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void	fconv_SSE(const cv::Mat &A, const cv::Mat &F, cv::Mat &R)
{
	int		RowA = A.rows, ColA = A.cols, NumFeatures = A.channels();
	int		RowF = F.rows, ColF = F.cols, ChnF = F.channels();
	if( NumFeatures!=ChnF || (NumFeatures%4) )
		throw runtime_error("");
	int loop = NumFeatures / 4;

	int		RowR = RowA - RowF + 1, ColR = ColA - ColF + 1;

	float *R_src0 = (float*)R.data;
	int Rstep = R.step1();

	const float	 *F_src = F.ptr<float>(0,0);
	const float	 *A_src0 = A.ptr<float>(0,0);

	__m128 a,b,c;
	for( int rr=0; rr<RowR; rr++ ){
		const float *A_src1 = A_src0 + rr*A.cols*NumFeatures; // start addr of A.row(rr)
		float *R_scr1 = R_src0 + rr*Rstep; // start addr of R.row(rr)
		for( int cc=0; cc<ColR; cc++ ){
			const float *A_src= A_src1 + cc*NumFeatures;// A.ptr<float>(rr,cc);			
			float *R_src = R_scr1 + cc;
			// An acceleration trick of using SSE programming >>>
			__m128 v = _mm_setzero_ps();
			const float *B_off = F_src;
			for( int rp=0; rp<RowF; rp++ ){
				const float *A_off = A_src + rp*A.cols*NumFeatures;
				for( int cp=0; cp<ColF; cp++ ){
					for( int l=0; l<loop; l++ ){
						a = _mm_load_ps(A_off);
						b = _mm_load_ps(B_off);
						c = _mm_mul_ps(a, b);
						v = _mm_add_ps(v, c);
						A_off += 4;
						B_off += 4;
					}
				}
			}
#ifdef WIN32
			*R_src =	 v.m128_f32[0] + v.m128_f32[1] + v.m128_f32[2] + v.m128_f32[3];
#else
			float buf[4] __attribute__ ((aligned (16)));
			_mm_store_ps(buf, v);
			_mm_empty();
			*R_src = buf[0]+buf[1]+buf[2]+buf[3];			
#endif
		}
	}
}