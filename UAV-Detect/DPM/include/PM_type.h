#ifndef PM_TYPE_H
#define PM_TYPE_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <limits>
#include <xmmintrin.h> // only for Intel processor

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;


#include "pthread.h"

#ifndef	MIN
#define MIN(x,y)	 (x<y?x:y)
#endif

#ifndef MAX
#define MAX(x,y)	 (x>y?x:y)
#endif

#ifndef INT_INF
//#define INT_INF		INT_MAX	// int32 ranges [-2147483648,2147483647]
// the c runtime file <limits> has defined INT_MAX & INT_MIN & FLT_MAX & FLT_MIN
#define INT_INF	(int)1e8
#endif

#ifndef FLOAT_INF
//#define FLOAT_INF	float(3e38/10000) // in fact, float32 ranges (4e38,4e38)
//#define FLOAT_INF	numeric_limits<float>::infinity() // 1.#INF 
#define FLOAT_INF	(float)1e20
#endif

namespace	PM_type
{
	
	template<class T, int num> struct AlignedArr
	{
		T x[num];
		AlignedArr() { memset(x,0,num*sizeof(T)); }
		AlignedArr( const AlignedArr *rhs ) { memcpy(x,rhs->x,num*sizeof(T)); }		
		void operator = ( const AlignedArr rhs ) { memcpy(x,rhs.x,num*sizeof(T)); }
		T & operator [] (int i) { return x[i]; }
		int size() { return num; }
	};

	typedef AlignedArr<int,2> INTS_2;
	typedef AlignedArr<int,3> INTS_3;
	typedef AlignedArr<int,4> INTS_4;
	typedef AlignedArr<float,2> FLOATS_2;
	typedef AlignedArr<float,4> FLOATS_4;
	typedef AlignedArr<float,7> FLOATS_7;

	const int SSS = 4; // deformation rules will be applied within the 4-elem proximity
	typedef AlignedArr<float,SSS*2+1> FLOATS_SSS;



	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	struct MODEL
	{
		string	Class, year, note;
		char type;
		int	numfilters, numblocks, numsymbols, sbin, interval, start;
		int	maxsize[2], minsize[2];
		float thresh;
		// for temporary use in some functions
		int			TheRoot;

		struct FILTERS{
			int		blocklabel;
			int		size[2];
			bool	flip;
			int		symbol;
		};
		vector<FILTERS>		filters;

		struct	RULES{
			char		type;
			int			i;
			vector<int>		lhs, rhs; // left hand symbols & right hand symbols
			int			detwindow[2], shiftwindow[2];
			vector<int>		blocks;
			struct{
				int		blocklabel;
			}offset;
			struct{
				int		blocklabel;
			}loc;
			struct{
				int		blocklabel;
				bool	flip;
			}def;
			vector<INTS_3>	anchor;
			// for temporary use in some functions
			vector<cv::Mat> Ix, Iy;
		};
		vector<vector<RULES> >		rules;

		struct SYMBOLS{
			char		type;
			int			filter;
			// for temporary use in some functions
			vector<cv::Mat>	score;
		};
		vector<SYMBOLS>	symbols;

		struct	BLOCKS{
			vector<float>		w;
			vector<float>		lb;
			float		learn, reg_mult;
			int			dim, shape[3];
			char		type;
		};
		vector<BLOCKS>	blocks;

		struct{
			int			sbin, dim, truncation_dim;
			bool		extra_octave;
			float		bias;
		}features;

		struct{
			vector<float>		slave_problem_time;
			vector<float>		data_mining_time;
			vector<float>		pos_latent_time;
			vector<float>		filter_usage;
		}stats;

		struct	BBOXPRED{
			cv::Mat	x1, y1, x2, y2;
		};
		vector<BBOXPRED>	bboxpred;

		//-----------------model-analysis-------------------------------------------
		vector<cv::Mat>		x_filters; // extracted filters from model's storage block
		struct RULE_DATA{
			RULES  rule; // the structural rule for each component model
			vector<int> sym_terminal;
			vector<int> sym_nonTerminal;
			vector<int> filter_index; // filter index for each terminal part symbol appeared in the component
			vector<float> model_bias; // model bias for each level
			vector<FLOATS_4> defcoeff; // deformation coefficients for each part
			vector<INTS_2> anchor; // anchor positions for each part					
			int numparts;	cv::Mat loc_w; float bias; int num_levels; // auxiliary
			// for temporary use in some functions
			vector<FLOATS_SSS> DXDEF;
			vector<FLOATS_SSS> DYDEF;
		};
		vector<RULE_DATA> RuleData;
		
	};

	// "block_types.m" : 'F'(HOG filter), 'P'(HOG PCA filter (used by cascade)), 'D'(Separable Quadratic Deformation), 'O'(Other: offets, ...)
	enum	{ block_types_Filter = 'F', block_types_PCAFilter = 'P', block_types_SepQuadDef = 'D', block_types_Other = 'O' };

	// "model_types.m:" : 'M'(Mixture of star models with latent orientation), 'G'(General grammar model (no specific structure))
	enum	{ model_types_MixStar = 'M', model_types_Grammar = 'G' };

	struct FEATURE_PYRAMID{
		vector<cv::Mat>	feat;
		vector<float>	scales;
		int		imsize[2];
		int		num_levels;
		int		padx, pady;
	};


	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	int			yuGetCurrentTime( char flag = 'S' );	
	void		yuSaveMat(string Name, const cv::Mat &mm);
	void		cvPrint(const cv::Mat &A, int chann);

	void		load_model( const string FileName, MODEL &model );	

	void		model_get_block(MODEL &model, MODEL::FILTERS &obj, cv::Mat &f);
	void		model_get_block(MODEL &model, int bl, cv::Mat &f);
	cv::Mat		model_get_block( MODEL &model, int bl, bool flip = false );	

	void		flipfeat(cv:: Mat &f, int method );
	void		cvPermutate( cv::Mat &f, int *new_row_idx, int *new_col_idx, int *new_dim_idx );		
	void		yuInd2Sub( const int Ind, const int *Sz, int &Row, int &Col, int &Dim );

	cv::Mat		features( const cv::Mat &image, const int sbin, const int pad[2] );	
	cv::Mat		features14_2( const cv::Mat &image, const int sbin, const int pad[2] );	

#define FEATURES PM_type::features
//#define FEATURES PM_type::features14_2

	void		featpyramid( const cv::Mat &im, MODEL &model, FEATURE_PYRAMID &pyra );
	void		featpyramid2( const cv::Mat &im, const MODEL &model, FEATURE_PYRAMID &pyra ); // backup

	void		analyze_model( MODEL &model );
	void		get_defcache( MODEL::RULE_DATA &RuleData, const int s );
	void		update_ruleData( MODEL &model, int num_levels );
	void		loc_feat( const MODEL &model, int num_levels, cv::Mat &f );

	void		fconv(const vector<cv::Mat> &Feats, const cv::Mat &Filter, vector<cv::Mat> &Response, int lev1, int lev2, vector<INTS_2> &ScoreSz);
	void		bounded_dt(const cv::Mat &vals, FLOATS_4 &defcoeff, cv::Mat &M, cv::Mat &Ix, cv::Mat &Iy);
	void		bounded_dt2(const cv::Mat &vals, FLOATS_SSS &DXDEF, FLOATS_SSS &DYDEF, cv::Mat &M, cv::Mat &Ix, cv::Mat &Iy);
	void		sum_score(cv::Mat &rootScore, const cv::Mat &partScore, int anchorx, int anchory, int padx, int pady);
	vector<INTS_2>	get_score_size(const vector<cv::Mat> &feats, const vector<cv::Mat> &filters);

	void		apply_rules( MODEL &model, const FEATURE_PYRAMID &pyra );
	void		apply_rules2( MODEL &model, FEATURE_PYRAMID &pyra );
	void		apply_rules3( MODEL &model, FEATURE_PYRAMID &pyra );

	void		detection_trees2( MODEL &model, int padx, int pady, vector<float>scales, 
					vector<int> Xv, vector<int> Yv, vector<int> Lv, vector<int> Cs, vector<float> Sv,
					cv::Mat &Dets, cv::Mat &Boxes);
	inline int pow2(int p) { return (1<<p); }	

	void		reduceboxes(const MODEL &model, cv::Mat &bs);
	void		bboxpred_get(const MODEL &model, cv::Mat &ds, const cv::Mat &bs);
	void		bboxpred_input(const cv::Mat &ds, const cv::Mat &bs, cv::Mat &A, cv::Mat &x1, cv::Mat &y1, cv::Mat &x2, cv::Mat &y2, cv::Mat &w, cv::Mat &h);
	void		clipboxes(const int imsize[2], cv::Mat &ds, cv::Mat &bs);
	cv::Mat		nms(const cv::Mat &boxes, float overlap);
	cv::Mat		pDollar_nms(const cv::Mat &bbs, float overlap);

	vector<FLOATS_7>	parse_detections( MODEL &model, FEATURE_PYRAMID &pyra, float thresh );
	void		draw_img(cv::Mat &img, vector<FLOATS_7> &detections, float ElapsedTime);

	
}
extern cv::Point2f	UL_INIT, BR_INIT;

#endif