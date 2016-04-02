#include "PM_type.h"
cv::Point2f	UL_INIT, BR_INIT;
void  PM_type::draw_img(cv::Mat &img, vector<FLOATS_7> &detections, float ElapsedTime)
{
	char buf[100];
	const cv::Scalar drawColor = CV_RGB(255, 0, 0);
	const cv::Scalar fontColor = CV_RGB(200, 0, 255);
	const float fontScale = 1;
	const int fontFace = CV_FONT_HERSHEY_PLAIN;

	printf(">>> %-9s %-9s %-9s %-9s %-5s %-9s %-9s\n","x1","y1","x2","y2","level","component","score");
	for( unsigned i=0; i<detections.size(); i++ ){
		float		x1 = detections[i][0], y1 = detections[i][1], x2 = detections[i][2], y2 = detections[i][3];
		float		level = detections[i][4];
		float		component = detections[i][5];		
		float		score = detections[i][6];

		cv::Point2f	UL(x1, y1), BR(x2, y2);
		if (i == 0)//draw only number 0
		{

			UL_INIT = UL, BR_INIT = BR;

			rectangle(img, UL, BR, drawColor, 2);
			printf("    %-9.3f %-9.3f %-9.3f %-9.3f %-5g %-9g %-9g\n", x1, y1, x2, y2, level, component, score);

			memset(buf, 0, 100);
			sprintf_s(buf, "%-2d %-3.0f %-3.0f %-3.0f %-3.0f  %-2.0f %-2.0f %-6.3f", i, x1, y1, x2, y2, level, component, score);
			string msg(buf);
			int	 baseline = 0;
			cv::Size textSize = cv::getTextSize(msg, fontFace, fontScale, 1, &baseline);
			cv::Point txtOrig(5, (i + 1)*(textSize.height + 3));
			putText(img, msg, txtOrig, fontFace, fontScale, fontColor);

			char buf2[4] = { '0' };
			sprintf_s(buf2, 4, "%d", i);
			msg = buf2;
			cv::Point textOrg2(int(x1) + 3, int(y1) + textSize.height + 5);
			putText(img, msg, textOrg2, fontFace, fontScale, drawColor);
		}

	
	}

	if( ElapsedTime>0 ){
		printf("Elapsed time for total program = %gs\n",ElapsedTime);
		memset( buf, 0, 100 );
		sprintf_s( buf, "elapsed_time = %gs", ElapsedTime );
		string msg(buf);	
		int	 baseline = 0;
		cv::Size textSize = cv::getTextSize(msg, fontFace, fontScale, 1, &baseline);
		cv::Point txtOrig(5, (detections.size() + 1)*(textSize.height + 3));
		putText( img, msg, txtOrig, fontFace, fontScale, fontColor );
	}		
}