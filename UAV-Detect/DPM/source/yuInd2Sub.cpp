#include "PM_type.h"

void PM_type::yuInd2Sub(const int Ind, const int *Sz, int &Row, int &Col, int &Dim)
{
	int	Area = Sz[0] * Sz[1];
	Dim = Ind / Area;
	int	Res = Ind - Dim * Area;
	Col = Res / Sz[0];
	Row = Res - Col*Sz[0];
}