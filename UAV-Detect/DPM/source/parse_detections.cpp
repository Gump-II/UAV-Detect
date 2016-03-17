#include "PM_type.h"

vector<PM_type::FLOATS_7> PM_type::parse_detections( MODEL &model, FEATURE_PYRAMID &pyra, float thresh )
{
	vector<FLOATS_7> Result;

	// Find scores above the threshold
	vector<int>	Xs, Ys, Ls;
	vector<float>	Ss;
	for(int level=model.interval; level<pyra.num_levels; level++ ){
		cv::Mat	&score = model.symbols[model.TheRoot].score[level];
		cv::Mat	Ax = score > thresh;
		for( int y=0; y<Ax.rows; y++ ){
			uchar *pt = Ax.ptr<uchar>(y,0);
			for( int x=0; x<Ax.cols; x++ ){
				if( *pt ){
					Xs.push_back(x);
					Ys.push_back(y);
					Ls.push_back(level);
					Ss.push_back( score.at<float>(y,x) );
				}
				pt++;
			}
		}
	}

	// Exit if no detection
	if(Xs.empty()) return Result;
	
	// Sort by scores
	vector<int>	ord( Xs.size() );
	cv::Mat	ord_mat(1, Xs.size(), CV_32SC1, &ord[0]);
	cv::Mat	Ss_mat(1, Ss.size(), CV_32FC1, &Ss[0]);
	sortIdx( Ss_mat, ord_mat, CV_SORT_EVERY_ROW+CV_SORT_DESCENDING );
	//
	vector<int>	tmpvec( ord.size() );
	for( unsigned i=0; i<ord.size(); i++ )
		tmpvec[i] = Xs[ ord[i] ];
	Xs = tmpvec;
	for( unsigned i=0; i<ord.size(); i++ )
		tmpvec[i] = Ys[ ord[i] ];
	Ys = tmpvec;
	for( unsigned i=0; i<ord.size(); i++ )
		tmpvec[i] = Ls[ ord[i] ];
	Ls = tmpvec;
	vector<float>tmpvecf( ord.size() );
	for( unsigned i=0; i<ord.size(); i++ )
		tmpvecf[i] = Ss[ ord[i] ];
	Ss = tmpvecf;

	// find the component of each score
	vector<int>	Cs;
	Cs.resize( Ss.size(), 0 );
	for( unsigned i=0; i<ord.size(); i++ ){
		float s = Ss[i];
		int lv = Ls[i], x = Xs[i], y = Ys[i];
		/* check */
		int rootSym0 = model.RuleData[0].sym_nonTerminal[0];
		cv::Mat &score0 = model.symbols[rootSym0].score[lv];
		if( score0.at<float>(y,x) != s ) 
			throw runtime_error(""); 
		for( unsigned component=1; component<model.RuleData.size(); component++ ){
			int rootSym = model.RuleData[component].sym_nonTerminal[0];
			cv::Mat &score = model.symbols[rootSym].score[lv];
			if( score.at<float>(y,x) >= s ){ Cs[i] = component; break; }
		}		
	}

	// Compute detection windows
	cv::Mat	ds, bs;
	detection_trees2( model, pyra.padx, pyra.pady, pyra.scales, Xs, Ys, Ls, Cs, Ss, ds, bs );
	clipboxes(pyra.imsize, ds, bs);
	bboxpred_get(model,ds,bs);
	cv::Mat numb;
	clipboxes(pyra.imsize, ds, numb);
	/* method 1 */
	cv::Mat dets = nms(ds, 0.5f);
	/* method 2 */
	//Mat dets = pDollar_nms( ds, 0.5f );

	int Cols = dets.cols;
	if( Cols!=7 )
		throw runtime_error("");
	Result.resize( dets.rows );
	float *pt = (float*)dets.data;
	for( int i=0; i<dets.rows; i++ ){
		for( int j=0; j<Cols; j++ )
			Result[i][j] = *(pt++);
	}

	return Result;
}