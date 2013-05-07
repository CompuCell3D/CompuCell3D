//---------------------------------------------------------------------------

#pragma hdrstop
#include <sstream>
#include "utils.h"
//---------------------------------------------------------------------------
#pragma package(smart_init)
using namespace std;
//---------------------------------------------------------------------------

string std_str(const string& str)
{
	return str;
}

string std_str(const char* str)
{
	return string(str);
}

string std_str(const String& str)
{
	return std_str(wstring(str.c_str()));
}

string std_str( const std::wstring& str )
{
	ostringstream stm;
	const ctype<char>& ctfacet =
						 use_facet< ctype<char> >( stm.getloc() );
	for( size_t i=0 ; i<str.size() ; ++i )
    {
		stm << ctfacet.narrow( str[i], 0 );
    }
	return stm.str() ;
}


