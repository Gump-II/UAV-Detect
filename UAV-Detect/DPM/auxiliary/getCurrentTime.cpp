 #include "PM_type.h"
#include "Windows.h"

int	 PM_type::yuGetCurrentTime( char flag )
{
	SYSTEMTIME	system;
	GetLocalTime( &system );
	int	seconds = system.wHour*3600 + system.wMinute*60 + system.wSecond;
	if( flag=='S' )	 return seconds;
	int Mseconds = 1000*seconds + system.wMilliseconds;
	return Mseconds;
}