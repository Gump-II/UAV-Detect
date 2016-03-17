#include "PM_type.h"

// the original implementation of featpyramid()

namespace PM_type{
	void	featpyramid2(const cv::Mat &im, MODEL &model, FEATURE_PYRAMID &pyra)
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
		// Amount to pad each level of the feature pyramid.
		// We pad the feature maps to detect partially visible objects.
		// padx    Amount to pad in the x direction
		// pady    Amount to pad in the y direction
		// Use the dimensions of the max over detection windows
		int		padx = model.maxsize[1];
		int		pady = model.maxsize[0];
		int		pads[2] = { pady + 1, padx+ 1 };

		int		extra_interval = 0;
		if( model.features.extra_octave )
			extra_interval = model.interval;

		int		sbin = model.sbin;
		int		interval = model.interval;
		float	sc = powf( 2.f, 1.f/interval );
		int		imsize[2] = { im.rows, im.cols };
		int		min_imsize = MIN( imsize[0],imsize[1] );
		int		max_scale = 1 + int(logf(min_imsize/(5.f*sbin))/logf(sc));

		int		SZ = max_scale + extra_interval + interval;
		//printf("SZ = %d, max_scale = %d, min_imsize = %d, imsize = %d %d\n",SZ,max_scale,min_imsize,imsize[0],imsize[1]);

		pyra.imsize[0] = imsize[0];
		pyra.imsize[1] = imsize[1];
		pyra.feat.resize( SZ );
		pyra.scales.resize( SZ, 0 );

		cv::Mat	im_float;
		if( im.depth()==CV_32F )
			im_float = im;
		else
			im.convertTo( im_float, CV_32FC3 );

		for( int i=0; i!=interval; i++ ){
			cv::Mat	scaled;
			float	scale_factor = 1.f / powf(sc,float(i));
			resize(im_float, scaled, cv::Size(), scale_factor, scale_factor, cv::INTER_AREA);
			if( extra_interval>0 ){
				// Optional (sbin/4) x (sbin/4) features
				//pyra.feat[i] = features( scaled, sbin/4, pads );
				pyra.feat[i] = FEATURES( scaled, sbin/4, pads );
				pyra.scales[i] = 4 * scale_factor;
			}
			// (sbin/2) x (sbin/2) features
			//pyra.feat[i+extra_interval] = features( scaled, sbin/2, pads );
			pyra.feat[i+extra_interval] = FEATURES( scaled, sbin/2, pads );
			pyra.scales[i+extra_interval] = 2 * scale_factor;
			// sbin x sbin HOG features 
			//pyra.feat[i+extra_interval+interval] = features( scaled, sbin, pads );
			pyra.feat[i+extra_interval+interval] = FEATURES( scaled, sbin, pads );
			pyra.scales[i+extra_interval+interval] = scale_factor;
			// Remaining pyramid octaves 
			for( int j=i+interval; j<max_scale; j+=interval ){			
				resize(scaled, scaled, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
				//pyra.feat[j+extra_interval+interval] = features( scaled, sbin, pads );
				pyra.feat[j+extra_interval+interval] = FEATURES( scaled, sbin, pads );
				pyra.scales[j+extra_interval+interval] = 0.5f * pyra.scales[j+extra_interval];
			}
		}

		pyra.num_levels = pyra.feat.size();
		pyra.padx = padx;
		pyra.pady = pady;
	}


	//	TEST
	// 
	//MODEL	modelt;
	//loadModel( "featpyra_model.txt", modelt);
	//Mat		img = imread( "featpyra.bmp" );
	//FEATURE_PYRAMID		pyramid;

	//featpyramid( img, modelt, pyramid );

	//FileStorage file_xml("pyramid.xml", FileStorage::WRITE);
	//file_xml<<"pyramid"<<pyramid.feat[5];
	//file_xml.release();
}