#include "PM_type.h"

void	PM_type::yuSaveMat(string Name, const cv::Mat &mm)
{
	string	Name2 = Name + ".xml";
	cv::FileStorage	fs(Name2, cv::FileStorage::WRITE);
	fs<<Name<<mm;
	fs.release();
}