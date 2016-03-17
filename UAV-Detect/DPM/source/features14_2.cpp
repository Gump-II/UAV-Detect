#include "PM_type.h"

// the 9+4+1=14 features only: 9 contrast-insensitive features, 4 texture features, 1 truncation feature
// finally, 14+2=16 features, the last two are zeros, for convenience of SSE programming
// takes a float color image and a bin size 
// returns HOG features
//Mat		features( const Mat &image, const int sbin );

//--------------------------------------------------------
// small value, used to avoid division by zero
#ifndef eps
#define eps	0.0001f
#endif

// unit vectors used to compute gradient orientation
const float	uu[9] = {1.0000f, 0.9397f, 0.7660f, 0.500f, 0.1736f, -0.1736f, -0.5000f, -0.7660f, -0.9397f};
const float	vv[9] = {0.0000f, 0.3420f, 0.6428f, 0.8660f, 0.9848f, 0.9848f, 0.8660f, 0.6428f, 0.3420f};
inline int yuSnap9_V1( float dx, float dy )
{
	// snap to one of 18 orientations
	float best_dot = 0;
	int best_o = 0;
	for (int o = 0; o < 9; o++) {
		float dot = uu[o]*dx + vv[o]*dy;
		if( dot<0 )
			dot = -dot;
		if (dot > best_dot) {
			best_dot = dot;
			best_o = o;
		}
	}
	return best_o;
}
inline int yuSnap9( float dx, float dy )
{
	// snap to 1 of the 9 orientations
	float a[4] = { uu[1]*dx, uu[2]*dx, uu[3]*dx, uu[4]*dx };
	float b[4] = { vv[1]*dy, vv[2]*dy, vv[3]*dy, vv[4]*dy };
	float dots[9] = { dx, a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3], b[3]-a[3], b[2]-a[2], b[1]-a[1], b[0]-a[0] }; // uu[i]*dx+vv[i]*dy
	float best_dot = 0;
	int best_o = 0;
	for( int i=0; i<9; i++ ){
		if( dots[i]<0 )
			dots[i] = -dots[i];
		if( dots[i]>best_dot ){
			best_dot = dots[i];
			best_o = i;
		}
	}
	return best_o;
}
inline int yuSnap9_V3( float dx, float dy )
{
	static const cv::Vec4f   a(uu[1], uu[2], uu[3], uu[4]);
	static const cv::Vec4f   b(vv[1], vv[2], vv[3], vv[4]);
	cv::Vec4f ux = a * dx;
	cv::Vec4f vy = b * dy;
	cv::Vec4f dot1 = ux + vy;
	cv::Vec4f dot2 = vy - ux;
	int best_o = 0;
	return best_o;
}

#ifndef	MIN
#define MIN(x,y)	 (x<y?x:y)
#endif
#ifndef MAX
#define MAX(x,y)	 (x>y?x:y)
#endif

// 因为featpyramid()中对各个pyra.feat[i]进行0/1扩展耗费了大量时间，为了减小这个开支，这里直接对计算的feature扩展
cv::Mat		PM_type::features14_2(const cv::Mat &image, const int sbin, const int pad[2])
{
	const int	dims[3] = { image.rows, image.cols, image.channels() };
	const int	row_step = image.cols * image.channels();
	const float * const imgpt = image.ptr<float>(0,0);
	if( image.type()!=CV_32FC3 )
		throw runtime_error("Invalid input");

	// memory for caching orientation histograms & their norms
	const int	blocks[2] = {  (int)(0.5f+(float)dims[0]/(float)sbin), (int)(0.5f+(float)dims[1]/(float)sbin) };
	const int	block_sz = blocks[0] * blocks[1];
	const int	visible[2] = { blocks[0]*sbin, blocks[1]*sbin };

	cv::Mat		histmat[9];
	float		*hist[9];	
	int			hist_step[9];
	cv::Mat		hist_tmp = cv::Mat::zeros(blocks[0], blocks[1] * 9, CV_32FC1);
	for( int i=0; i<9; i++ ){
		//histmat[i] = Mat::zeros( blocks[0], blocks[1], CV_32FC1 );
		histmat[i] = hist_tmp(cv::Rect(i*blocks[1], 0, blocks[1], blocks[0]));
		hist[i] = histmat[i].ptr<float>(0,0);
		hist_step[i] = histmat[i].step1();
	}

	// 1. Calculate the original HOG
	for (int yy = 1; yy < visible[0]-1; yy++) { // each row
		int		y = MIN(yy,dims[0]-2);
		const float	*imgpt1 = imgpt + y*row_step;
		for (int xx = 1; xx < visible[1]-1; xx++) { // each col
			int		x = MIN(xx,dims[1]-2);

			// b-g-r color at (y,x) is { *s, *(s+1), *(s+2) }
			const float	*s = imgpt1 + x*dims[2];
			float Right[3] = { *(s+3), *(s+4), *(s+5) }; 
			float Left[3] = { *(s-3), *(s-2), *(s-1) };
			float Up[3] = { *(s-row_step), *(s+1-row_step), *(s+2-row_step) };
			float Down[3] = { *(s+row_step), *(s+1+row_step), *(s+2+row_step) };

			// gradients and amplitude of orientations of each channel
			float	dxs, dys, vs;
			float	dx, dy, v=-1;
			for( int i=0; i!=3; i++ ){
				dxs = Right[i] - Left[i];
				dys = Down[i] - Up[i];
				vs = dxs*dxs + dys*dys;
				if( vs>=v ){
					v = vs;
					dx = dxs;
					dy = dys;
				}
			}

			//int best_o = yuSnap9_V1( dx, dy );
			//int best_o = yuSnap9( dx, dy );
			int best_o = 0;
			{
				// snap to 1 of the 9 orientations
				float a[4] = { uu[1]*dx, uu[2]*dx, uu[3]*dx, uu[4]*dx };
				float b[4] = { vv[1]*dy, vv[2]*dy, vv[3]*dy, vv[4]*dy };
				float dots[9] = { dx, a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3], b[3]-a[3], b[2]-a[2], b[1]-a[1], b[0]-a[0] }; // uu[i]*dx+vv[i]*dy
				float best_dot = 0;
				for( int i=0; i<9; i++ ){
					if( dots[i]<0 )
						dots[i] = -dots[i];
					if( dots[i]>best_dot ){
						best_dot = dots[i];
						best_o = i;
					}
				}
			}

			// add to 4 histograms around pixel using linear interpolation
			float xp = (xx+0.5f)/(float)sbin - 0.5f;
			float yp = (yy+0.5f)/(float)sbin - 0.5f;
			//int ixp = (int)floorf(xp),  iyp = (int)floorf(yp);
			int ixp = (int)xp, iyp = (int)yp;
			if( xp<0 && xp!=ixp )
				ixp--;
			if( yp<0 && yp!=iyp )
				iyp--;
			int ixp_p = ixp + 1, iyp_p = iyp + 1;
			float vx0 = xp-ixp, vy0 = yp-iyp;
			float vx1 = 1.f-vx0, vy1 = 1.f-vy0;
			v = sqrtf(v);

			if( ixp<0 ){
				ixp = 0;
				vx1 = 0;
			}
			if( ixp_p>=blocks[1] ){
				ixp_p = blocks[1] - 1;
				vx0 = 0;
			}
			if( iyp<0 ){
				iyp = 0;
				vy1 = 0;
			}
			if( iyp_p>=blocks[0] ){
				iyp_p = blocks[0] - 1;
				vy0 = 0;
			}

			float	*hist_pt = hist[best_o] + iyp*hist_step[best_o] + ixp;
			ixp_p = ixp_p - ixp;
			iyp_p = iyp_p - iyp;
			*(hist_pt) += vx1*vy1*v;
			*(hist_pt + ixp_p) += vx0*vy1*v;
			*(hist_pt + iyp_p*hist_step[best_o]) += vx1*vy0*v;
			*(hist_pt + iyp_p*hist_step[best_o] + ixp_p) += vx0*vy0*v;
		}
	}

	// 2. Compute energy in each block by summing over orientations
	cv::Mat		normmat = cv::Mat::zeros(blocks[0], blocks[1], CV_32FC1);
	float * const norm = normmat.ptr<float>(0,0);
	const int	norm_step = normmat.step1();
	for( int i=0; i<9; i++ )
		accumulateSquare( histmat[i], normmat );

	// normalization coefficients
	cv::Mat	Normmss(blocks[0] - 1, blocks[1] - 1, CV_32FC1);
	float * const Norms = Normmss.ptr<float>(0,0);
	const int	norms_step = Normmss.step1();
	for( int i=0; i<Normmss.rows; i++ ){
		float	*src = norm + i*norm_step;
		float	*dst = Norms + i*norms_step;
		for( int j=0; j<Normmss.cols; j++ ){
			float	tmp = *src + *(src+1) + *(src+norm_step) + *(src+norm_step+1) + eps;
			//*(dst++) = 1.f / sqrtf(tmp);
			*(dst++) = tmp;
			src++;
		}
	}
	cv::sqrt( Normmss, Normmss );
	Normmss = 1.f / Normmss;

	// 3. Reduced HOG features.
	int out[3];
	out[0] = MAX(blocks[0]-2, 0);
	out[1] = MAX(blocks[1]-2, 0);
	out[2] = 9+4+1; // 14 dimensional features
	out[2] += 2; // 16 dimensional features finally
	cv::Mat	feat_mat(out[0] + 2 * pad[0], out[1] + 2 * pad[1], CV_32FC(out[2]));
	cv::Vec<float, 16>	init_val;
	init_val = 0;
	init_val[13] = 1;
	feat_mat.setTo( init_val );
	float * const feat_data = feat_mat.ptr<float>(pad[0],pad[1]);
	const int feat_step = feat_mat.step1();

	for( int y=0; y<out[0]; y++ ){
		float	*dst = feat_data + y*feat_step;
		float	*Normpt = Norms + y*norms_step;
		for( int x=0; x<out[1]; x++ ){
			// 			
			float	n4 = *Normpt;
			float	n2 = *(Normpt+1);
			float	n3 = *(Normpt+norms_step);
			float	n1 = *(Normpt+norms_step+1);
			Normpt++;

			float	t1 = 0, t2 = 0, t3 = 0, t4 = 0;

			// 9 contrast-insensitive features
			for( int i=0; i<9; i++ ){
				float	src = *( hist[i]+(y+1)*hist_step[i]+x+1 );
				//float	src = histmat_insens[i].at<float>(y+1,x+1);
				float h1 = MIN(src * n1, 0.2f);
				float h2 = MIN(src * n2, 0.2f);
				float h3 = MIN(src * n3, 0.2f);
				float h4 = MIN(src * n4, 0.2f);
				*(dst++) = 0.5f * (h1 + h2 + h3 + h4);
				t1 += h1;
				t2 += h2;
				t3 += h3;
				t4 += h4;
			}

			// 4 texture features
			*(dst++) = 0.3333f * t1;
			*(dst++) = 0.3333f * t2;
			*(dst++) = 0.3333f * t3;
			*(dst++) = 0.3333f * t4;

			// 1 truncation feature	 and 2 supplementary features
			//dst += 3;
			*(dst++) = 0;
			*(dst++) = 0;
			*(dst++) = 0;

		}
	}

	return	feat_mat;
}

/* 如何加速

1. 使用宏（MIN,MAX）比使用内联函数要快
2. 一次分配数据空间比多次分配数据空间要快：
a. 多个同样大小的矩阵--一次分配一个大的空间，每个矩阵分别索引该空间的一个子块
b. 函数内部使用的中间矩阵，如果大小不变，可以声明静态变量；如果大小会变，但是有上线，可以使用静态vector
3. 标准库模板stl中提供了一些常用函数，如 floorf(),sqrtf()，但是这些函数的效率不高，经常使用会导致函数性能下降，
如果可以，尽量使用整型截取(int())来代替floorf，使用openCV提供的cv::sqrt函数来一次处理大量数据的sqrtf()操作。
另外，std::sqrtf()函数要比std::sqrt()函数要快（在大量重复试验下，可以看出明显差异）。
4. 尽可能使用指针操作--指针操作是最快速内存访问方法，要避免使用openCV提供的矩阵元素获取的函数。

*/