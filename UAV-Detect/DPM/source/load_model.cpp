 #include "PM_type.h"


inline
void yuCheck( bool val )
{
	if( !val ){
		printf("Check failure in loadModel()! Please check the model file.\n");
		throw runtime_error("");
	}
}

inline
ifstream& operator>>( ifstream &ifs, int &val )
{
	// infinite numbers can be written into file with "inf" markup( both in matlab and in c++ ),
	// but when reading from the file, the "inf" markup is not recognized as a number
	string	tmpS;
	ifs>>tmpS;
	if( tmpS=="Inf" || tmpS=="inf" )
		val = INT_INF;
	else if( tmpS=="-Inf" || tmpS=="-inf" )
		val = -INT_INF;
	else
		val = (int)atoi( tmpS.c_str() );

	return	ifs;	
}

inline
ifstream& operator>>( ifstream &ifs, float &val )
{
	string	tmpS;
	ifs>>tmpS;
	if( tmpS=="Inf" || tmpS=="inf" )
		val = FLOAT_INF;
	else if( tmpS=="-Inf" || tmpS=="-inf" )
		val = -FLOAT_INF;
	else
		val = (float)atof( tmpS.c_str() );

	return	ifs;
}

//=============================================
void PM_type::load_model( const string FileName, MODEL &model )
// load a DPM model from the file
{
	ifstream	MF( FileName.c_str() );
	if( !MF ){
		cout<<"Cannot open model file!"<<endl;
		throw runtime_error("");
	}

	string	Title, tmpS;
	int	Sz, tmp;

	// class
	MF>>Title;
	yuCheck(Title=="class");
	MF>>model.Class;
	// year
	MF>>Title;
	yuCheck(Title=="year");
	MF>>model.year;
	// note
	MF>>Title;
	yuCheck(Title=="note");
	MF>>model.note;

	// filters
	MF>>Title;
	yuCheck(Title=="filters");
	MF>>Sz;
	model.filters.resize( Sz );
	for( unsigned i=0; i<model.filters.size(); i++ ){
		MODEL::FILTERS	F;
		MF>>tmpS;
		MF>>Title;
		yuCheck(Title=="blocklabel");
		MF>>F.blocklabel;
		MF>>Title;
		yuCheck(Title=="size");
		MF>>F.size[0]>>F.size[1];
		MF>>Title;
		yuCheck(Title=="flip");
		MF>>tmp;
		F.flip = (tmp!=0);
		MF>>Title;
		yuCheck(Title=="symbol");
		MF>>F.symbol;
		// -1
		F.blocklabel--;
		F.symbol--;
		model.filters[i] = F;
	}

	// rules
	MF>>Title;
	yuCheck(Title=="rules");
	MF>>Sz;
	model.rules.resize( Sz );
	for( unsigned i=0; i<model.rules.size(); i++ ){		
		MF>>tmpS;
		MF>>Sz;
		model.rules[i].resize( Sz );
		for( unsigned j=0; j<model.rules[i].size(); j++ ){
			MODEL::RULES		R;
			MF>>tmpS;
			MF>>Title;
			yuCheck(Title=="type");
			MF>>R.type;
			MF>>Title;
			yuCheck(Title=="lhs");
			MF>>Sz;
			R.lhs.resize( Sz );
			for( unsigned k=0; k<R.lhs.size(); k++ )
				MF>>R.lhs[k];
			MF>>Title;
			yuCheck(Title=="rhs");
			MF>>Sz;
			R.rhs.resize( Sz );
			for( unsigned k=0; k<R.rhs.size(); k++ )
				MF>>R.rhs[k];
			MF>>Title;
			yuCheck(Title=="detwindow");
			MF>>R.detwindow[0]>>R.detwindow[1];
			MF>>Title;
			yuCheck(Title=="shiftwindow");
			MF>>R.shiftwindow[0]>>R.shiftwindow[1];
			MF>>Title;
			yuCheck(Title=="i");
			MF>>R.i;
			MF>>Title;
			yuCheck(Title=="anchor");
			MF>>Sz;
			R.anchor.resize( Sz );
			for( unsigned k=0; k<R.anchor.size(); k++ ){
				MF>>tmp;
				yuCheck(tmp==k+1);
				MF>>R.anchor[k][0]>>R.anchor[k][1]>>R.anchor[k][2];
			}
			MF>>Title;
			yuCheck(Title=="offset.blocklabel");
			MF>>R.offset.blocklabel;
			MF>>Title;
			yuCheck(Title=="loc.blocklabel");
			MF>>R.loc.blocklabel;
			MF>>Title;
			yuCheck(Title=="blocks");
			MF>>Sz;
			R.blocks.resize( Sz );
			for( unsigned k=0; k<R.blocks.size(); k++ )
				MF>>R.blocks[k];
			MF>>Title;
			yuCheck(Title=="def.blocklabel");
			MF>>R.def.blocklabel;
			MF>>Title;
			yuCheck(Title=="def.flip");
			MF>>tmp;
			R.def.flip = (tmp!=0);
			// -1
			for( unsigned k=0; k<R.lhs.size(); k++ )
				R.lhs[k]--;
			for( unsigned k=0; k<R.rhs.size(); k++ )
				R.rhs[k]--;
			R.i--;
			R.offset.blocklabel--;
			R.loc.blocklabel--;
			R.def.blocklabel--;
			model.rules[i][j] = R;
		}
	}

	// symbols
	MF>>Title;
	yuCheck(Title=="symbols");
	MF>>Sz;
	model.symbols.resize( Sz );
	for( unsigned i=0; i<model.symbols.size(); i++ ){
		MODEL::SYMBOLS	SS;
		MF>>tmpS;
		MF>>Title;
		yuCheck(Title=="type");
		MF>>SS.type;
		MF>>Title;
		yuCheck(Title=="filter");
		MF>>SS.filter;
		// -1
		SS.filter--;
		model.symbols[i] = SS;
	}

	// numfilters
	MF>>Title;
	yuCheck(Title=="numfilters");
	MF>>model.numfilters;
	MF>>Title;
	yuCheck(Title=="numblocks");
	MF>>model.numblocks;
	MF>>Title;
	yuCheck(Title=="numsymbols");
	MF>>model.numsymbols;
	MF>>Title;
	yuCheck(Title=="start");
	MF>>model.start;
	MF>>Title;
	yuCheck(Title=="maxsize");
	MF>>model.maxsize[0]>>model.maxsize[1];
	MF>>Title;
	yuCheck(Title=="minsize");
	MF>>model.minsize[0]>>model.minsize[1];
	MF>>Title;
	yuCheck(Title=="interval");
	MF>>model.interval;
	MF>>Title;
	yuCheck(Title=="sbin");
	MF>>model.sbin;
	MF>>Title;
	yuCheck(Title=="thresh");
	MF>>model.thresh;
	MF>>Title;
	yuCheck(Title=="type");
	MF>>model.type;
	// -1
	model.start--;

	// blocks
	MF>>Title;
	yuCheck(Title=="blocks");
	MF>>Sz;
	model.blocks.resize( Sz );
	for( unsigned i=0; i<model.blocks.size(); i++ ){
		MODEL::BLOCKS	B;
		MF>>tmpS;
		MF>>Title;
		yuCheck(Title=="w");
		MF>>Sz;
		B.w.resize( Sz );
		for( int k=0; k<Sz; k++ )
			MF>>B.w[k];
		MF>>Title;
		yuCheck(Title=="lb");
		MF>>Sz;
		B.lb.resize( Sz );
		for( int k=0; k<Sz; k++ )
			MF>>B.lb[k];
		MF>>Title;
		yuCheck(Title=="learn");
		MF>>B.learn;
		MF>>Title;
		yuCheck(Title=="reg_mult");
		MF>>B.reg_mult;
		MF>>Title;
		yuCheck(Title=="dim");
		MF>>B.dim;
		MF>>Title;
		yuCheck(Title=="shape");
		MF>>Sz;
		B.shape[0] = B.shape[1] = B.shape[2] = 1;
		for( int k=0; k<Sz; k++)
			MF>>B.shape[k];
		MF>>Title;
		yuCheck(Title=="type");
		MF>>B.type;
		// 
		model.blocks[i] = B;
	}

	// features
	MF>>Title;
	yuCheck(Title=="features");
	MF>>Title;
	yuCheck(Title=="sbin");
	MF>>model.features.sbin;
	MF>>Title;
	yuCheck(Title=="dim");
	MF>>model.features.dim;
	MF>>Title;
	yuCheck(Title=="truncation_dim");
	MF>>model.features.truncation_dim;
	MF>>Title;
	yuCheck(Title=="extra_octave");
	MF>>tmp;
	model.features.extra_octave = (tmp!=0);
	MF>>Title;
	yuCheck(Title=="bias");
	MF>>model.features.bias;

	// stats
	MF>>Title;
	yuCheck(Title=="stats");
	MF>>Title;
	yuCheck(Title=="slave_problem_time");
	MF>>Sz;
	model.stats.slave_problem_time.resize( Sz );
	for( int i=0; i<Sz; i++ )
		MF>>model.stats.slave_problem_time[i];
	MF>>Title;
	yuCheck(Title=="data_mining_time");
	MF>>Sz;
	model.stats.data_mining_time.resize( Sz );
	for( int i=0; i<Sz; i++ )
		MF>>model.stats.data_mining_time[i];
	MF>>Title;
	yuCheck(Title=="pos_latent_time");
	MF>>Sz;
	model.stats.pos_latent_time.resize( Sz );
	for( int i=0; i<Sz; i++ )
		MF>>model.stats.pos_latent_time[i];
	MF>>Title;
	yuCheck(Title=="filter_usage");
	MF>>Sz;
	model.stats.filter_usage.resize( Sz );
	for( int i=0; i<Sz; i++ )
		MF>>model.stats.filter_usage[i];

	// bboxpred
	MF>>Title;
	yuCheck(Title=="bboxpred");
	MF>>Sz;
	model.bboxpred.resize( Sz );
	for( unsigned i=0; i<model.bboxpred.size(); i++ ){
		MODEL::BBOXPRED		B2;
		MF>>tmpS;
		MF>>Title;
		yuCheck(Title=="x1");
		MF>>Sz;
		B2.x1.create( Sz, 1, CV_32FC1 );
		float *pt = (float*)B2.x1.data;
		for( int k=0; k<Sz; k++ )
			MF>>pt[k];
		MF>>Title;
		yuCheck(Title=="y1");
		MF>>Sz;
		B2.y1.create( Sz, 1, CV_32FC1 );
		pt = (float*)B2.y1.data;
		for( int k=0; k<Sz; k++ )
			MF>>pt[k];
		MF>>Title;
		yuCheck(Title=="x2");
		MF>>Sz;
		B2.x2.create( Sz, 1, CV_32FC1 );
		pt = (float*)B2.x2.data;
		for( int k=0; k<Sz; k++ )
			MF>>pt[k];
		MF>>Title;
		yuCheck(Title=="y2");
		MF>>Sz;
		B2.y2.create( Sz, 1, CV_32FC1 );
		pt = (float*)B2.y2.data;
		for( int k=0; k<Sz; k++ )
			MF>>pt[k];
		model.bboxpred[i] = B2;
	}


	MF.close();
	return;

}