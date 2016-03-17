#ifndef FAST_DPM_H
#define FAST_DPM_H

#include "PM_type.h"
using namespace PM_type;

class FastDPM
{
public:
	FastDPM( string model_file );

	cv::Mat prepareImg(cv::Mat &img_uint8); // returns a float type image and set "img_color" with appropriate value	

	void detect(cv::Mat &img, float score_thresh = -FLOAT_INF, bool show_hints = true, bool show_img = true, string save_img = "");
	vector<FLOATS_7>	detections; // results

	//void	show( Mat &the_img, bool specification=false );

	/* All useful stuff have been declared above. */
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	/* Below are not supposed to be used. */

public:
	MODEL		model;	
	FEATURE_PYRAMID		pyra;

	cv::Mat		img_color;

	bool		hints;
	int			start_clock;
	int			end_clock;
	int			prag_start;
	int			prag_end;

};

#endif