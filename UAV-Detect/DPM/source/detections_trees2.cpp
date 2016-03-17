#include "PM_type.h"
using namespace PM_type;

void PM_type::detection_trees2( MODEL &model, int padx, int pady, vector<float>scales, 
								 vector<int> Xv, vector<int> Yv, vector<int> Lv, vector<int> Cs, vector<float> Sv,
								 cv::Mat &Dets, cv::Mat &Boxes)
{
	int dim = Sv.size();
	Dets = cv::Mat::zeros(dim, 4 + 3, CV_32FC1); // 4+3 is bounding box, level, component, score
	float *dets = (float*)Dets.data;
	int numparts = model.RuleData[0].numparts;
	Boxes = cv::Mat::zeros(dim, 4 * (1 + numparts), CV_32FC1);
	float *boxes = (float*)Boxes.data; 
	//
	const float sbin = (float)model.sbin;
	for( int i=0; i<dim; i++ ){
		const int x = Xv[i];
		const int y = Yv[i];
		const int lv = Lv[i];
		const int comp = Cs[i];
		float score = Sv[i];
		MODEL::RULE_DATA &ruleData = model.RuleData[comp];
		// structural rule
		int sym = model.start;
		MODEL::RULES &rule = model.rules[sym][comp];
		float scale = sbin / scales[lv];
		float x1 = ( x - rule.shiftwindow[1] - padx )*scale;
		float y1 = ( y - rule.shiftwindow[0] - pady )*scale;
		float x2 = x1 + rule.detwindow[1]*scale - 1;
		float y2 = y1 + rule.detwindow[0]*scale - 1;
		*(dets++) = x1; *(dets++) = y1; *(dets++) = x2; *(dets++) = y2; 
		*(dets++) = (float)lv; *(dets++) = (float)comp; *(dets++) = score;
		// deformation rule - root symbol
		int filter_index = ruleData.filter_index[0];
		x1 = ( x - padx ) * scale;
		y1 = ( y - pady ) * scale;
		x2 = x1 + model.x_filters[filter_index].cols * scale - 1;
		y2 = y1 + model.x_filters[filter_index].rows * scale - 1;
		*(boxes++) = x1; *(boxes++) = y1; *(boxes++) = x2; *(boxes++) = y2;
		// deformation rule - part symbols
		int partlv = lv - model.interval;
		scale = sbin / scales[partlv];
		for( unsigned idx=1; idx<rule.rhs.size(); idx++ ){			
			int ax = ruleData.anchor[idx-1][0];
			int ay = ruleData.anchor[idx-1][1];
			int partx = x*2 + ax;
			int party = y*2 + ay;		
			partx -= padx; party -= pady;
			sym = rule.rhs[idx];
			cv::Mat	&Ix = model.rules[sym][0].Ix[partlv];
			cv::Mat	&Iy = model.rules[sym][0].Iy[partlv];
			int rhs_x = Ix.at<int>(party,partx);
			int rhs_y = Iy.at<int>(party,partx);
			//			
			filter_index = ruleData.filter_index[idx];
			x1 = ( rhs_x - padx ) * scale;
			y1 = ( rhs_y - pady ) * scale;
			x2 = x1 + model.x_filters[filter_index].cols * scale - 1;
			y2 = y1 + model.x_filters[filter_index].rows * scale - 1;
			*(boxes++) = x1; *(boxes++) = y1; *(boxes++) = x2; *(boxes++) = y2;
		}
	}
	return;
}