#include "PM_type.h"

struct getFeat_data{
	vector<cv::Mat> *MatArr;
	PM_type::FEATURE_PYRAMID *pyra;
	int interval, level, len;
	int *pads, *binSz;
};

void * getFeat( void *p )
{
	getFeat_data *D = (getFeat_data*)p;
	int level = D->level;
	int interval = D->interval;
	int *binSz = D->binSz;
	int len = D->len;
	int *pads = D->pads;
	vector<cv::Mat> &feat = D->pyra->feat;
	vector<cv::Mat> *MatArr = D->MatArr;
	//printf("thread-%d begins\n",level);
	for( int i=0; i<len; i++ )
		//feat[level+i] = PM_type::features( (*MatArr)[level+i], binSz[level+i], pads );
		//feat[level+i] = PM_type::features14_2( (*MatArr)[level+i], binSz[level+i], pads );
		feat[level+i] = FEATURES( (*MatArr)[level+i], binSz[level+i], pads );
	//printf("thread-%d ends\n",level);
	return NULL;
}

void PM_type::featpyramid(const cv::Mat &im, MODEL &model, FEATURE_PYRAMID &pyra)
//% Compute a feature pyramid.
//%   pyra = featpyramid(im, model, padx, pady)
//%
//% Return value
//%   pyra    Feature pyramid (see details below)
//%
//% Arguments
//%   im      Input image
//%   model   Model (for use in determining amount of 
//				   %           padding if pad{x,y} not given)
//				   %   padx    Amount of padding in the x direction (for each level)
//				   %   pady    Amount of padding in the y direction (for each level)
//				   %
//				   % Pyramid structure (basics)
//				   %   pyra.feat{i}    The i-th level of the feature pyramid
//				   %   pyra.feat{i+interval} 
//%                   Feature map computed at exactly half the 
//%                   resolution of pyra.feat{i}
{
	if( im.type()!=CV_32FC3 ) throw runtime_error("");
	// Amount to pad each level of the feature pyramid.
	// We pad the feature maps to detect partially visible objects.
	// padx    Amount to pad in the x direction
	// pady    Amount to pad in the y direction
	// Use the dimensions of the max over detection windows
	int		padx = model.maxsize[1];
	int		pady = model.maxsize[0];
	int		pads[2] = { pady + 1, padx+ 1 };

	// I think the extra_interval is useless for that the DPM is weaker in detecting small objects, extra calc is unnecessary.
	// if( model.features.extra_octave ) extra_interval = model.interval;	

	int		sbin = model.sbin;
	int		interval = model.interval;
	float	sc = powf( 2.f, 1.f/interval );
	int		imsize[2] = { im.rows, im.cols };
	int		min_imsize = MIN( imsize[0],imsize[1] );
	int		max_scale = 1 + int(logf(min_imsize/(5.f*sbin))/logf(sc));
	int		SZ = max_scale + interval;

	pyra.imsize[0] = imsize[0];
	pyra.imsize[1] = imsize[1];
	pyra.feat.resize( SZ );
	pyra.scales.resize( SZ );
	vector<int> BinSz( SZ, sbin );

	// resized images for each level
	static vector<cv::Mat> MatArr;
	MatArr.resize( SZ );
	// the second octave
	MatArr[interval] = im; pyra.scales[interval] = 1.f;
	sc = 1.f/sc; float scf = sc;for( int i=1; i<interval; i++ )
	{
		resize(MatArr[interval], MatArr[interval + i], cv::Size(), scf, scf, cv::INTER_AREA); pyra.scales[interval + i] = scf; scf *= sc;
	}
	// the first octave
	for( int i=0; i<interval; i++ ) { MatArr[i] = MatArr[i+interval]; pyra.scales[i] = pyra.scales[interval+i] * 2.f; BinSz[i] = sbin/2; }
	// the rest octaves	
	int octaves = SZ / interval + 1;
	for( int oc=2; oc<octaves; oc++ ){
		for( int i=0; i<interval; i++ ){
			int lv = oc*interval+i; if( lv>=SZ ) break;
			pyrDown( MatArr[lv-interval], MatArr[lv] ); pyra.scales[lv] = pyra.scales[lv-interval] * 0.5f;
		}		
	}

	// Create threads to process each level features
	//getFeat_data data; data.MatArr = &MatArr; data.interval = interval; data.pads = pads; data.sbin = sbin; data.pyra = &pyra;
	getFeat_data data[2]; 
	for( int i=0; i<2; i++ ){ data[i].MatArr = &MatArr; data[i].interval = interval; data[i].pads = pads; data[i].binSz = &BinSz[0]; data[i].pyra = &pyra; }
	data[0].level = 0; data[0].len = 5; 
	data[1].level = data[0].len; data[1].len = 8;
	pthread_t	threads[2];
	for( int i=0; i<2; i++ ) pthread_create( threads+i, NULL, getFeat, &(data[i]) );

	// Process the rest levels in the main thread
	for( int i=data[1].level+data[1].len; i<SZ; i++ )
		//pyra.feat[i] = features14_2( MatArr[i], sbin, pads );
		//pyra.feat[i] = features( MatArr[i], sbin, pads );
		pyra.feat[i] = FEATURES( MatArr[i], sbin, pads );
	//printf("main-thread ends\n");

	pyra.num_levels = pyra.feat.size();
	pyra.padx = padx;
	pyra.pady = pady;

	for( int i=0; i<2; i++ ) pthread_join( threads[i], NULL );

	return;
}