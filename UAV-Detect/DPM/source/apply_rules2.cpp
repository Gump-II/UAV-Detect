#include "PM_type.h"
using namespace PM_type;

/*
filter_responses:
Use each filter( model.filters ) to filter the detection image's feature-pyramid( array of feature matrices ) respectively.

The response of filter[i] on feature-pyramid's specific level is stored in 

model.symbols[a].score[level], where. a = model.filters[i].symbol
*/

namespace APPLY_RULES2{

	struct{
		MODEL *model;
		FEATURE_PYRAMID *pyra;
		vector<INTS_2> ScoreSz;
	} ThreadRules;

	struct ThreadParam{
		int comp;
		int filterIndex;
	};

	void * thread_each_part( void *p )
	{
		MODEL *model = ThreadRules.model;
		const FEATURE_PYRAMID *pyra = ThreadRules.pyra;

		ThreadParam *pData = (ThreadParam*)p;
		int comp = pData->comp;
		MODEL::RULE_DATA &RuleData = model->RuleData[comp];
		int f = pData->filterIndex; 
		//printf("\n>>thread(%d) begins",f);

		int partFilterIndex = RuleData.filter_index[f];
		int partSymbol = RuleData.sym_terminal[f];
		int partNonTerminal = RuleData.sym_nonTerminal[f];
		int num_levels = pyra->num_levels, interval = model->interval;

		// filter response of each part ( including root part )
		const vector<cv::Mat> &filters = model->x_filters;
		vector<cv::Mat> &SymScore = model->symbols[partSymbol].score;
		SymScore.resize( num_levels );
		if( f==0 )
			fconv( pyra->feat, filters[partFilterIndex], SymScore, interval, num_levels, ThreadRules.ScoreSz );
		else
			fconv( pyra->feat, filters[partFilterIndex], SymScore, 0, num_levels-interval, ThreadRules.ScoreSz );

		if( f==0 ){
			//printf("\n>>thread(%d) ends",f);
			return NULL;
		}

		// apply deformation rule
		vector<cv::Mat> &SymNtScore = model->symbols[partNonTerminal].score;
		vector<cv::Mat> &Ix = model->rules[partNonTerminal][0].Ix;
		vector<cv::Mat> &Iy = model->rules[partNonTerminal][0].Iy;
		SymNtScore.resize( num_levels );
		Ix.resize( num_levels );
		Iy.resize( num_levels );
		for( int lv=0; lv<num_levels-interval; lv++ )
			/* method 1. */
			//bounded_dt( SymScore[lv], RuleData.defcoeff[f-1], SymNtScore[lv], Ix[lv], Iy[lv] );
			/* method 2. */
			bounded_dt2( SymScore[lv], RuleData.DXDEF[f-1], RuleData.DYDEF[f-1], SymNtScore[lv], Ix[lv], Iy[lv] ); 

		//printf("\n>>thread(%d) ends",f);
		return NULL;
	}
}

using namespace APPLY_RULES2;


void	PM_type::apply_rules2( MODEL &model, FEATURE_PYRAMID &pyra )
{
	const vector<cv::Mat>	&filters = model.x_filters;
	vector<INTS_2>		ScoreSz = get_score_size( pyra.feat, filters );

	for( unsigned component=0; component<model.RuleData.size(); component++ ){
		MODEL::RULE_DATA &RuleData = model.RuleData[component];
		int numThreads = RuleData.filter_index.size();
		ThreadRules.model = &model;
		ThreadRules.pyra = &pyra;
		ThreadRules.ScoreSz = ScoreSz;
		vector<pthread_t> Threads( numThreads );
		vector<ThreadParam> Ps( numThreads );
		for( int f=0; f<numThreads; f++ ){
			Ps[f].comp = component;
			Ps[f].filterIndex = f;
			pthread_create( &Threads[f], NULL, thread_each_part, &Ps[f] );
		}
		for( int f=0; f<numThreads; f++ )
			pthread_join( Threads[f], NULL );
		// apply structural rule
		int rootSymbol = RuleData.sym_terminal[0];
		int rootNonTerminal = RuleData.sym_nonTerminal[0];
		model.symbols[rootNonTerminal].score.resize( pyra.num_levels );
		for( int lv=model.interval; lv<pyra.num_levels; lv++ ){
			cv::Mat &score = model.symbols[rootNonTerminal].score[lv];
			score = model.symbols[rootSymbol].score[lv] + cv::Scalar(RuleData.model_bias[lv]);
			int partlv = lv - model.interval;
			for( int part=0; part<RuleData.numparts; part++ ){
				cv::Mat &pscore = model.symbols[RuleData.sym_nonTerminal[part + 1]].score[partlv];
				sum_score( score, pscore, RuleData.anchor[part][0], RuleData.anchor[part][1], pyra.padx, pyra.pady );		
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