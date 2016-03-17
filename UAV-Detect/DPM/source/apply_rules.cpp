#include "PM_type.h"

/*
filter_responses:
Use each filter( model.filters ) to filter the detection image's feature-pyramid( array of feature matrices ) respectively.

The response of filter[i] on feature-pyramid's specific level is stored in 

model.symbols[a].score[level], where. a = model.filters[i].symbol
*/

void	PM_type::apply_rules( MODEL &model, const FEATURE_PYRAMID &pyra )
{
	const vector<cv::Mat>	&filters = model.x_filters;
	vector<INTS_2>		ScoreSz = get_score_size( pyra.feat, filters );

	for( unsigned component=0; component<model.RuleData.size(); component++ ){
		MODEL::RULE_DATA &RuleData = model.RuleData[component];
		// root filter response
		int rootFilterIndex = RuleData.filter_index[0];
		int rootSymbol = RuleData.sym_terminal[0];
		model.symbols[rootSymbol].score.resize( pyra.num_levels );
		fconv( pyra.feat, filters[rootFilterIndex], model.symbols[rootSymbol].score, model.interval, pyra.num_levels, ScoreSz );		
		// part filter response
		for( unsigned f=1; f<RuleData.filter_index.size(); f++ ){
			int partFilterIndex = RuleData.filter_index[f];
			int partSymbol = RuleData.sym_terminal[f];
			model.symbols[partSymbol].score.resize( pyra.num_levels );
			fconv( pyra.feat, filters[partFilterIndex], model.symbols[partSymbol].score, 0, pyra.num_levels-model.interval, ScoreSz );			
			// apply deformation rule
			int partNonTerminal = RuleData.sym_nonTerminal[f];
			model.symbols[partNonTerminal].score.resize( pyra.num_levels );
			model.rules[partNonTerminal][0].Ix.resize( pyra.num_levels );
			model.rules[partNonTerminal][0].Iy.resize( pyra.num_levels );
			for( int lv=0; lv<pyra.num_levels-model.interval; lv++ )
				/* method 1. */
				//bounded_dt( model.symbols[partSymbol].score[lv], RuleData.defcoeff[f-1],  \
				//model.symbols[partNonTerminal].score[lv], model.rules[partNonTerminal][0].Ix[lv], model.rules[partNonTerminal][0].Iy[lv] );
				/* method 2. */
				bounded_dt2( model.symbols[partSymbol].score[lv], RuleData.DXDEF[f-1], RuleData.DYDEF[f-1], \
				model.symbols[partNonTerminal].score[lv], model.rules[partNonTerminal][0].Ix[lv], model.rules[partNonTerminal][0].Iy[lv] );
		}
		// apply structural rule
		int rootNonTerminal = RuleData.sym_nonTerminal[0];
		model.symbols[rootNonTerminal].score.resize( pyra.num_levels );
		for( int lv=model.interval; lv<pyra.num_levels; lv++ ){
			cv::Mat &score = model.symbols[rootNonTerminal].score[lv];
			score = model.symbols[rootSymbol].score[lv] + cv::Scalar(RuleData.model_bias[lv]);
			int partlv = lv - model.interval;
			for( int part=0; part<RuleData.numparts; part++ ){
				cv::Mat &pscore = model.symbols[RuleData.sym_nonTerminal[part + 1]].score[partlv];
				/* method 1. */
				sum_score( score, pscore, RuleData.anchor[part][0], RuleData.anchor[part][1], pyra.padx, pyra.pady );
				/* method 2. */
				//int step = 2;
				//int startx = RuleData.anchor[part][0] - pyra.padx;
				//int starty = RuleData.anchor[part][1] - pyra.pady;									
				//int endx = MIN( pscore.cols-1, startx+step*(score.cols-1) ); // then length(starty:step:endy)==score[i].cols
				//int endy = MIN( pscore.rows-1, starty+step*(score.rows-1) );
				//vector<int>	ix, iy; 	int	 oy = 0, ox = 0;
				//for(int tmpy=starty;tmpy<=endy;tmpy+=step)if( tmpy<0)oy++;else iy.push_back(tmpy);
				//for(int tmpx=startx;tmpx<=endx;tmpx+=step)if(tmpx<0)ox++;else ix.push_back( tmpx );
				//Mat	sp( iy.size(), ix.size(), CV_32FC1 );
				//for( int k1=0; k1!=sp.rows; k1++ )	
				//	for( int k2=0; k2!=sp.cols; k2++ )
				//		sp.at<float>(k1,k2) = pscore.at<float>(iy[k1],ix[k2]);
				//Mat	stmp( score.rows, score.cols, CV_32FC1, Scalar(-FLOAT_INF) );
				//sp.copyTo( stmp(Rect(ox,oy,sp.cols,sp.rows)) ); 
				//score = score + stmp;				
			}
		}
	}

	// symbol_score: apply the very structural rule of this mixture star model
	int rootSym1 = model.RuleData[0].sym_nonTerminal[0];
	for( unsigned component=1; component<model.RuleData.size(); component++ ){		
		int rootSym2 = model.RuleData[component].sym_nonTerminal[0];
		for( int lv=model.interval; lv<pyra.num_levels; lv++ ){
			cv::Mat &score1 = model.symbols[rootSym1].score[lv];
			cv::Mat &score2 = model.symbols[rootSym2].score[lv];
			score1 = cv::max( score1, score2 );
		}
	}
	// store results: 
	/* method 1. */
	model.TheRoot = rootSym1;
	/* method 2. only copy the Mat headers */
	//model.symbols[model.start].score.resize( pyra.num_levels );
	//for( int lv=model.interval; lv<pyra.num_levels; lv++ )
	//	model.symbols[model.start].score[lv] = model.symbols[rootSym1].score[lv];

	return;
}