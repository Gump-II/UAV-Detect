#include <string>
#include <vector>
#include <set>
#include <iostream>
#include <fstream>
#include <io.h>
#include <algorithm>


using namespace std;

// Read file names of specified name extensions in the given dir.
vector<string>  yuStdDirFiles( string DirName, vector<string> FileExtensions );
// Split absolute file path into director+filename+extension.
void  yuSplitFileNames( const vector<string> &AbsoNamePaths, vector<string> &DirNames, vector<string> &FileNames, vector<string> &ExtenedNames );
// Find the common part (prefix & suffix) in a list of strings.
string yuFindPrefix( const vector<string> &strs );
string yuFindSuffix( const vector<string> &strs );
// if FLAG=0, return the file name without path but has extension;
// if FLAG=1, return the file name without path and extension;
// if FLAG=2, return the extension name.
string trim_file_name( string FileName, int FLAG );
// Copy file (text or image file, etc.).
void yuCopyFile( const char *src, const char *dst );

// some else directory manipulations with standard c library:
// _mkdir( const char * );
// remove( const char * );

///////////////////////////////////////////
vector<string>  yuStdDirFiles( string DirName, vector<string> FileExtensions )
// Read file names of specified name-extensions  in the given dir
// DirName can be: "C:\\WINDOWS" or "C:\\WINDOWS\\" and such like
// "." means current dir; "../" means the upper dir.
{
	set<string> Extensions;
	bool READ_ALL = FileExtensions.empty();
	if( !READ_ALL )
		Extensions.insert( FileExtensions.begin(), FileExtensions.end() );

	vector<string> FileNames;
	cout<<"Finding files in \""<<DirName<<"\":"<<endl;

	_finddata_t file;
	string DirName2 = DirName + "/*.*";
	long lf = _findfirst( DirName2.c_str(), &file );
	if( lf==-1l ){
		cout<<"ERROR: INCORRECT DIRNAME: "<<DirName<<endl;
		throw runtime_error("ERROR: INCORRECT DIRNAME!");
	}
	char FullPath[512];
	_fullpath( FullPath, DirName.c_str(), 512 );
	string DirPath( FullPath );
	for( unsigned i=0; i<DirPath.size(); i++ ){
		if( DirPath[i]=='\\' )
			DirPath[i] = '/';
	}
	DirPath += "/";
	while( _findnext( lf, &file ) == 0 ){
		if( file.attrib==_A_SUBDIR || file.attrib==_A_SYSTEM || file.attrib==_A_HIDDEN ) // 子目录、系统文件、隐藏文件
			continue;
		string Name = file.name;
		string ExtenedName = trim_file_name( Name, 2 );
		if( Extensions.find(ExtenedName)==Extensions.end() )
			continue;
		Name = DirPath + Name;
		FileNames.push_back(Name);
	}

	cout<<"Have found "<<FileNames.size()<<" files (with given name extensions):"<<endl;
	int min_num = std::min( FileNames.size(), 5u );
	for( int i=0; i<min_num; i++ )
		cout<<"   "<<trim_file_name(FileNames[i],0)<<endl;
	cout<<"-------------- DONE --------------"<<endl;
	return FileNames;
}
///////////////////////////////////////////
// A small but useful function to modify a image file name to better display
// FLAG=0: return the file name without path, but has extension; 
// FLAG=1: return the file name without path and extension;
// FLAG=2: return the extension name.
string trim_file_name( string FileName, int FLAG )
{
	string::size_type p1 = FileName.rfind('/');
	string::size_type p2 = FileName.find('.');
	bool flag1 = p1==string::npos;
	bool flag2 = p2==string::npos;
	string ret = FileName;
	switch(FLAG){
	case 0:
		if(!flag1)
			ret = FileName.substr(p1+1);
		break;
	case 1:
		if(!flag1 && !flag2)
			ret = FileName.substr( p1+1, p2-p1 );
		else if( !flag1 )
			ret = FileName.substr( p1+1 );
		else if( !flag2 )
			ret = FileName.substr( 0, p2 );
		break;
	case 2:
		if(!flag2)
			ret = FileName.substr( p2 );
		else
			ret.clear();
		break;
	default:
		throw runtime_error("ERROR: INVALID FLAG!");
	}
	return ret;
}
///////////////////////////////////////////
// str1 & str2 has a common prefix whose length is the return value
unsigned findPrefix(const string &str1, const string &str2)
{
	unsigned i = 0;
	unsigned total = std::min(str1.size(), str2.size());
	for( i=0; i<total; i++ )
		if( str1[i]!=str2[i] )
			break;
	return i;
}
///////////////////////////////////////////
// str1 & str2 has a common suffix whose length is the return value
unsigned findSufffix(const string &str1, const string &str2)
{
	unsigned i = 0;
	unsigned total = std::min(str1.size(), str2.size());
	for( i=0; i<total; i++ ){
		unsigned j = total - i - 1;
		if( str1[j]!=str2[j] )
			break;
	}
	return i;
}
///////////////////////////////////////////
void  yuSplitFileNames(const vector<string> &AbsoNamePaths, vector<string> &DirNames, vector<string> &FileNames, vector<string> &ExtenedNames )
// AbsoNamePaths is a list of names with absolute file path. Function will split it into dir, file_name, extension
// e.g. if AbsoNamePaths = "C:/Program Files/haha.c", then DirNames = "C:/Program Files/", FileNames = "haha", ExtenedNames = ".c"
{
	int num = AbsoNamePaths.size();
	DirNames.resize( num );
	FileNames = ExtenedNames = DirNames;
	for( int i=0; i<num; i++ ){
		string Name = AbsoNamePaths[i];
		string::size_type p = Name.rfind('/');
		if( p!=string::npos ){
			DirNames[i] = Name.substr(0,p+1);
			Name.erase(0,p+1);
		}
		p = Name.rfind('.');
		if( p!=string::npos ){
			ExtenedNames[i] = Name.substr(p);
			Name.erase(p);
		}
		FileNames[i] = Name;
	}
}
///////////////////////////////////////////
// find the common prefix of a list of strings
string yuFindPrefix(const vector<string> &strs)
{
	string Prefix = strs[0];
	for( unsigned i=0; i<strs.size(); i++ ){
		unsigned a = findPrefix( Prefix, strs[i] );
		if( a==0 ){
			Prefix.clear();
			break;
		}
		if( a==Prefix.size() )
			continue;
		Prefix = Prefix.substr( 0, a );
	}
	return Prefix;
}
///////////////////////////////////////////
// find the common suffix of a list of strings
string yuFindSuffix(const vector<string> &strs)
{
	string Suffix = strs[0];
	for(unsigned i=0; i<strs.size(); i++){
		unsigned a = findSufffix( Suffix, strs[i]);
		if(a==0){
			Suffix.clear();
			break;
		}
		if(a==Suffix.size())
			continue;
		Suffix = Suffix.substr(Suffix.size()-a, a);
	}
	return Suffix;
}
///////////////////////////////////////////
// copy file
void yuCopyFile(const char *src, const char *dst)
{
	int buf[1024];
	FILE *pr,*pw;
	pr = fopen(src, "rb");
	pw = fopen(dst, "wb");
	while(!feof(pr)){
		fread(buf,1,1,pr);
		fwrite(buf,1,1,pw);}
	fclose(pr);
	fclose(pw);
}