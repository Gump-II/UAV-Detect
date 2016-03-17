#include "PM_type.h"

namespace PM_type{
	template <typename T>
	void	cvPrint(const cv::Mat &A, T type, int chann)
	{
		int		Rows = A.rows, Cols = A.cols, Chns = A.channels();
		int		A_ROW = Cols * Chns;
		if( chann>=0 && chann<Chns )
			A_ROW = Cols;	// 只打印特定通道
		else{
			Chns = 1; // 打印所有通道
			chann = 0;
		}
		for( int i=0; i!=Rows; i++ ){
			cout<<"\n------------------"<<i<<"th row-------------------------"<<endl;
			T		*ptr = (T*)(A.data) + i*A.step1(0);
			for( int j=0; j!=A_ROW; j++ )
				cout<<(float)ptr[j*Chns+chann]<<" ";		
		}
		cout<<endl;
	}

	void	cvPrint(const cv::Mat &A, int chann)
	{
		int		Depth = A.depth();
		switch( Depth ){
		case CV_8U:
			cvPrint( A, (uchar)0, chann );
			break;
		case CV_32S:
			cvPrint( A, (int)0, chann );
			break;
		case CV_32F:
			cvPrint( A, (float)0, chann );
			break;
		case CV_64F:
			cvPrint( A, (double)0, chann );
			break;
		default:
			cout<<"Not supported data type in cvPrint() !"<<endl;
			throw runtime_error("");
		}			
	}
}