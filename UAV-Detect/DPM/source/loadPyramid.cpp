#include "PM_type.h"
using namespace PM_type;

inline
void yuCheck( bool val )
{
	if( !val ){
		printf("Check failure!\n");
		throw runtime_error("");
	}

}

void loadPyramid( const string FileName, FEATURE_PYRAMID &pyra )
// 从文件中加载特征金字塔
{
	ifstream	MF( FileName.c_str() );
	if( !MF ){
		cout<<"Cannot open model file!"<<endl;
		throw runtime_error("");
	}

	string	Title, tmpS;
	int		Sz, tmp;

	// imsize
	MF>>Title;
	yuCheck(Title=="imsize");
	MF>>pyra.imsize[0]>>pyra.imsize[1];

	// num_levels
	MF>>Title;
	yuCheck(Title=="num_levels");
	MF>>pyra.num_levels;

	// padx
	MF>>Title;
	yuCheck(Title=="padx");
	MF>>pyra.padx;

	// pady
	MF>>Title;
	yuCheck(Title=="pady");
	MF>>pyra.pady;

	// scales
	MF>>Title;
	yuCheck(Title=="scales");
	MF>>Sz;
	pyra.scales.resize( Sz );
	for( int i=0; i<Sz; i++ )
		MF>>pyra.scales[i];

	// valid_levels
	MF>>Title;
	yuCheck(Title=="valid_levels");
	MF>>Sz;
	for( int i=0; i<Sz; i++ )
		MF>>tmp;

	// feat
	MF>>Title;
	yuCheck(Title=="feat");
	MF>>Sz;
	pyra.feat.resize( Sz );
	for( int i=0; i<Sz; i++ ){
		int		Rows, Cols, Dims;
		MF>>tmpS>>Rows>>Cols>>Dims;
		yuCheck(Dims==32);
		pyra.feat[i].create( Rows, Cols, CV_32FC(Dims) );
		float	*pt = pyra.feat[i].ptr<float>(0,0);
		for( int k=0; k<Rows*Cols*Dims; k++ )
			MF>>pt[k];
		MF>>tmpS;
		yuCheck(tmpS=="END");
	}
}