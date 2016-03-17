#include "PM_type.h"

void PM_type::analyze_model( MODEL &model )
{
	if( model.numfilters!=model.filters.size() )
		throw runtime_error("");
	model.x_filters.resize( model.numfilters );
	for( int i=0; i!=model.numfilters; i++ )
		model_get_block( model, model.filters[i], model.x_filters[i] );
	//
	int comps = model.rules[0].size();
	model.RuleData.resize( comps );
	for( int component=0; component<comps; component++ ){
		MODEL::RULE_DATA &RuleData = model.RuleData[component];
		RuleData.rule = model.rules[0][component];
		RuleData.numparts = RuleData.rule.rhs.size()-1;
		RuleData.sym_terminal.resize( RuleData.numparts+1 );
		RuleData.sym_nonTerminal.resize( RuleData.numparts+1 );
		RuleData.filter_index.resize( RuleData.numparts+1 );
		RuleData.defcoeff.resize( RuleData.numparts );
		RuleData.anchor.resize( RuleData.numparts );
		for( unsigned i=0; i<RuleData.rule.rhs.size(); i++ ){
			// the first is root terminal (root filter), the rest are part terminals (part filters)
			int symbol = RuleData.rule.rhs[i];
			int symbol_terminal = symbol - 1;
			RuleData.sym_terminal[i] = symbol_terminal;
			RuleData.sym_nonTerminal[i] = symbol;
			RuleData.filter_index[i] = model.symbols[symbol_terminal].filter;
			if( i==0 ){
				for( unsigned anc=0; anc<RuleData.anchor.size(); anc++ ){
					RuleData.anchor[anc][0] = RuleData.rule.anchor[anc+1][0];
					RuleData.anchor[anc][1] = RuleData.rule.anchor[anc+1][1];
				}
				cv::Mat	biasM = model_get_block(model, RuleData.rule.offset.blocklabel);
				RuleData.bias = biasM.at<float>(0)  * model.features.bias;
				RuleData.loc_w = model_get_block( model, RuleData.rule.loc.blocklabel );
			}
			else{
				MODEL::RULES &r = model.rules[symbol][0];
				cv::Mat defcoeff = model_get_block(model, r.def.blocklabel, r.def.flip);
				float *pt = defcoeff.ptr<float>(0);
				for( int j=0; j<4; j++ )
					RuleData.defcoeff[i-1][j] = *(pt++);
			}
		}
		get_defcache( RuleData, PM_type::SSS );
	}
}

void PM_type::update_ruleData( MODEL &model, int num_levels )
{
	int comps = model.RuleData.size();
	for( int component=0; component<comps; component++ ){
		MODEL::RULE_DATA &RuleData = model.RuleData[component];
		RuleData.num_levels = num_levels;
		cv::Mat	loc_f; loc_feat(model, num_levels, loc_f);
		cv::Mat	loc_scores; loc_scores = RuleData.loc_w * loc_f;
		cv::Mat	model_bias = loc_scores + cv::Scalar(RuleData.bias);
		float *pt = model_bias.ptr<float>(0);
		RuleData.model_bias.resize( num_levels );
		for( int i=0; i<num_levels; i++ )
			RuleData.model_bias[i] = *(pt++);
	}
}

void PM_type::get_defcache( MODEL::RULE_DATA &RuleData, const int s )
{
	vector<FLOATS_4> &defcoeff  = RuleData.defcoeff;
	int sz = defcoeff.size();
	RuleData.DXDEF.resize( sz );
	RuleData.DYDEF.resize( sz );
	for( int i=0; i<sz; i++ ){
		float ax = defcoeff[i][0], bx = defcoeff[i][1], ay = defcoeff[i][2], by = defcoeff[i][3];
		for( int dist=-s; dist<=s; dist++ ){
			int j = dist + s;
			RuleData.DXDEF[i][j] = -ax*dist*dist - bx*dist;
			RuleData.DYDEF[i][j] = -ay*dist*dist - by*dist;
		}
	}
	//ofstream fd("dxdef");
	//for( int i=0; i<RuleData.DXDEF.size(); i++ ){
	//	for( int j=0; j<RuleData.DXDEF[0].size(); j++ )
	//		fd<<RuleData.DXDEF[i][j]<<" ";
	//	fd<<"\n";
	//}
	//fd<<"-----------"<<endl;
	//for( int i=0; i<RuleData.DYDEF.size(); i++ ){
	//	for( int j=0; j<RuleData.DYDEF[0].size(); j++ )
	//		fd<<RuleData.DYDEF[i][j]<<" ";
	//	fd<<"\n";
	//}
	//fd<<endl;
	//fd.close();
}