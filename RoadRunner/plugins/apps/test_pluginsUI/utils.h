#ifndef utilsH
#define utilsH
#include <System.hpp>
#include <string>

using std::string;
using std::wstring;
//---------------------------------------------------------------------------
string std_str( const char* str );
string std_str( const string& str );
string std_str( const wstring& str );
string std_str( const String& str );

#endif
