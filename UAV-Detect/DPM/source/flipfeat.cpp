#include "PM_type.h"

void PM_type::flipfeat(cv::Mat &f, int method)
{
	if( f.channels()!=16 && f.channels()!=32 ){
		printf( "WRONG IN filpfeat() : f.channels()==%d\n", f.channels() );
		throw runtime_error("");
	}

	// There is the "flip()" function in openCV that may help to accelerate.
	//static int cnt = 1;
	//printf("flipfeat(%d)\t",cnt++);

	int p1[16] = { 
		0, 8, 7, 6, 5, 4, 3, 2, 1,\
		11, 12, 9, 10,\
		13,\
		14, 15
	};

	int p2[32] = { 
		9,     8,     7,     6,     5,     4,     3,     2,     1,     0,    17,    16,    15,    14,    13,    12,
		11,  10,   18,   26,   25,   24,   23,    22,   21,   20,    19,    29,    30,    27,    28,    31
	};

	static vector<int>	rc;
	if( rc.size()!=f.cols ){
		rc.resize( f.cols );
		for( int i=0; i!=rc.size(); i++ )
			rc[i] = rc.size() - 1 - i;
	}

	int *p = p1;
	if( f.channels()==32 ) 
		p = p2;

	if( method!=0 )
		cvPermutate( f, NULL, &rc[0], NULL );
	else
		cvPermutate( f, NULL, &rc[0], p );
}