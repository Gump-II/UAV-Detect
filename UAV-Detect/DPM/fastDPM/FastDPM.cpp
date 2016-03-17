#include "FastDPM.h"


FastDPM::FastDPM( string model_file )
{
	// 1.
	load_model( model_file, model );
	analyze_model( model );
}

cv::Mat FastDPM::prepareImg(cv::Mat &img_uint8)
// Make a uint8 type image into a float type mat with 3 channels
{
	if( img_uint8.depth()!=CV_8U || (img_uint8.channels()!=1 && img_uint8.channels()!=3) ){
		printf("Function prepareImg() only takes as input an image of 1 or 3 channels of uint8 type!");
		throw	runtime_error("");
	}
	if( img_uint8.channels()==1 )
		cvtColor( img_uint8, img_color, CV_GRAY2RGB );
	else if( img_uint8.channels()==3 )
		img_color = img_uint8;
	cv::Mat	img;
	img_color.convertTo( img, CV_32FC3 );
	return	img;
}

void	loadPyramid( const string FileName, FEATURE_PYRAMID &pyra );
void FastDPM::detect(cv::Mat &img, float score_thresh /* = -FLOAT_INF */, bool show_hints /* = true */, bool show_img /* = true */, string save_img /* = */)
{
	hints = show_hints;
	detections.clear();
	prag_start = yuGetCurrentTime('M');
	// 1. Feature pyramid <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<	
	if( hints ){ printf("Calculating feature pyramid ...\t"); start_clock = prag_start; }
	/* method 1. */
	featpyramid( img, model, pyra );
	//loadPyramid( "pyramid.txt", pyra );
	/* method 2. */
	//featpyramid2( img, model, pyra );	
	if( hints ){	end_clock = yuGetCurrentTime('M');	printf("_featpyramid takes %gs\n",(end_clock-start_clock)/1000.f);	}

	update_ruleData( model, pyra.num_levels );

	// 2. Filter responses & Deformation rules & Structural rules<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	if( hints ){	printf("Applying rules ...\t"); start_clock = end_clock; }
	/* method 1. */
	//apply_rules( model, pyra );
	/* method 2. */
	//apply_rules2( model, pyra );
	/* method 3. */
	apply_rules3( model, pyra );
	if( hints ){	end_clock = yuGetCurrentTime('M');	printf("_apply_rules takes %gs\n",(end_clock-start_clock)/1000.f); }

	// 3. Parse detections <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	if( hints ){	printf("Parsing detections ...\t"); start_clock = end_clock; }
	detections = parse_detections( model, pyra, score_thresh );
	if( detections.empty() ) { printf("NO DETECTIONS!\n");  }// add 'return¡¯don't show no target
	if( hints ){	end_clock = yuGetCurrentTime('M');	printf("_parse_detections takes %gs\n",(end_clock-start_clock)/1000.f); }

	// 4. Show results <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	float Elapsed_Time = 0;
	if( hints ) { prag_end = end_clock; Elapsed_Time = (prag_end-prag_start)/1000.f; }
	/*if( show_img || !save_img.empty() ){
		draw_img( img_color, detections, Elapsed_Time );
		if( show_img )
			imshow( "FAST_DPM", img_color );
		if( !save_img.empty() )
			imwrite( save_img, img_color );
	}*/

	draw_img( img_color, detections, Elapsed_Time );

	//imshow( "FAST_DPM", img_color );

	
}

