#include "PM_type.h"

void PM_type::cvPermutate(cv::Mat &f, int *new_row_idx, int *new_col_idx, int *new_dim_idx)
{
	int	Rows = f.rows, Cols = f.cols, Chns = f.channels();

	// task can be down in 3 steps:

	// 1st step: f = f(:,:,new_dim_idx);	
	if( new_dim_idx!=NULL ){
		int		*from_to = new int [Chns*2]();
		for( int i=0; i!=Chns; i++ ){
			from_to[2*i] = new_dim_idx[i];
			from_to[2*i+1] = i;
		}
		cv::Mat		g = f.clone();
		mixChannels( &g, 1, &f, 1, from_to, Chns );
		g.release();
	}	

	// 2nd step: f = f(:,new_col_idx,:);
	if( new_col_idx!=NULL ){
		cv::Mat		f2(Rows, Cols*Chns, CV_MAKETYPE(f.depth(), 1), f.data);
		cv::Mat		g2 = f2.clone();
		for( int i=0; i!=Cols; i++ ){
			int			f_col_start = i * Chns;
			int			f_col_end = f_col_start + Chns;
			int			g_col_start = new_col_idx[i] * Chns;
			int			g_col_end = g_col_start + Chns;
			cv::Mat		f_col_block = f2.colRange(f_col_start, f_col_end);
			cv::Mat		g_col_block = g2.colRange(g_col_start, g_col_end);
			g_col_block.copyTo( f_col_block );
		}
		g2.release();
	}

	// 3rd step: f = f(new_row_idx,:,:);
	if( new_row_idx!=NULL ){
		cv::Mat	g3 = f.clone();
		for( int i=0; i!=Rows; i++ ){
			cv::Mat	f_row = f.row(i);
			cv::Mat	g_row = g3.row(new_row_idx[i]);
			g_row.copyTo( f_row );
		}
		g3.release();
	}

}