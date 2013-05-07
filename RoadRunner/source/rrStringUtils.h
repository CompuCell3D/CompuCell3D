#ifndef rrStringUtilsH
#define rrStringUtilsH
#include <string>
#include <list>
#include <vector>
#include "rrConstants.h"
#include "rrExporter.h"

namespace rr
{
using std::string;
using std::list;
using std::vector;

RR_DECLSPEC char*   			createText(const string& str);
RR_DECLSPEC string              ReplaceWord(const string& str1, const string& str2, const string& theString);
RR_DECLSPEC bool                ConvertFunctionCallToUseVarArgsSyntax(const string& funcName, string& expression);
RR_DECLSPEC string              RemoveChars(const string& str, const string& chars);
RR_DECLSPEC bool                IsUnwantedChar(char ch); //Predicate for find_if algorithms..
RR_DECLSPEC size_t              FindMatchingRightParenthesis(const string& expression, const size_t startFrom);
RR_DECLSPEC int                 GetNumberOfFunctionArguments(const string& expression);
RR_DECLSPEC string              tabs(const int& nr);
RR_DECLSPEC string              NL();

RR_DECLSPEC string				ToUpperOrLowerCase(const string& inStr, int (*func)(int));
RR_DECLSPEC string 				ToUpper(const string& str);
RR_DECLSPEC string 				ToLower(const string& str);

RR_DECLSPEC string              ExtractFilePath(const string& fileN);
RR_DECLSPEC string              ExtractFileName(const string& fileN);
RR_DECLSPEC string              ExtractFileNameNoExtension(const string& fileN);

RR_DECLSPEC string              ChangeFileExtensionTo(const string& theFileName, const string& newExtension);

RR_DECLSPEC int 				CompareNoCase(const string& str1, const string& str2);
RR_DECLSPEC string              Trim(const string& str, const char& toTrim = ' ');
RR_DECLSPEC bool                StartsWith(const string& src, const string& sub);
RR_DECLSPEC bool                EndsWith(const string& src, const string& sub);

//Can't use va_arg for non pod data.. :(
RR_DECLSPEC string              JoinPath(const string& p1, const string& p2, const char pathSeparator = gPathSeparator);
RR_DECLSPEC string              JoinPath(const string& p1, const string& p2, const string& p3, const char pathSeparator = gPathSeparator);
RR_DECLSPEC string              JoinPath(const string& p1, const string& p2, const string& p3, const string& p4, const char pathSeparator = gPathSeparator);
RR_DECLSPEC string              JoinPath(const string& p1, const string& p2, const string& p3, const string& p4, const string& p5, const char pathSeparator = gPathSeparator);

//conversions
RR_DECLSPEC string              IntToStr(const int& nt);
RR_DECLSPEC int                 StrToInt(const string& nt);
RR_DECLSPEC string              DblToStr(const double& nt);
RR_DECLSPEC double              StrToDbl(const string& nt);
RR_DECLSPEC vector<string>      SplitString(const string &text, const string &separators);
RR_DECLSPEC vector<string>      SplitString(const string& input, const char& delimiters);
RR_DECLSPEC int                 ToInt(const string& str);
RR_DECLSPEC bool                ToBool(const string& str);
RR_DECLSPEC double              ToDouble(const string& str);


RR_DECLSPEC string              ToString(const bool& b);
RR_DECLSPEC string              ToString(const double& d, const string& format = gDoubleFormat);
RR_DECLSPEC string              ToString(const unsigned int& n, const string& format = gIntFormat, const int nBase=10);
RR_DECLSPEC string              ToString(const int& n, const string& format = gIntFormat, const int nBase=10);
RR_DECLSPEC string              ToString(const long n, const int nBase=10);
RR_DECLSPEC string              ToString(const unsigned long n, const int nBase=10);
RR_DECLSPEC string              ToString(const unsigned short n, const int nBase=10);
RR_DECLSPEC string              ToString(const short n, const int nBase=10);
RR_DECLSPEC string              ToString(const char n);
RR_DECLSPEC string              ToString(const unsigned char n);
RR_DECLSPEC string              ToString(const string & s);
RR_DECLSPEC string              ToString(const char* str);

RR_DECLSPEC string              Format(const string& src, const int& arg);
RR_DECLSPEC string              Format(const string& str, const int& arg1);
RR_DECLSPEC string              Format(const string& src, const string& arg);
RR_DECLSPEC string              Format(const string& src, const string& arg1, const string& arg2, const string& arg3);
RR_DECLSPEC string              Format(const string& src, const string& arg1, const string& arg2);
RR_DECLSPEC string              Format(const string& src, const string& arg1, const int& arg2);
RR_DECLSPEC string              Format(const string& src, const string& arg1, const int& arg2, const string& arg3);
RR_DECLSPEC string              Format(const string& str1, const string& str2);
RR_DECLSPEC string              Format(const string& str1, const string& arg1, const string& arg2);
RR_DECLSPEC string              Format(const string& str1, const string& arg1, const int& arg2);
RR_DECLSPEC string              Format(const string& str1, const string& arg1, const string& arg2, const string& arg3);
RR_DECLSPEC string              Format(const string& str1, const string& arg1, const string& arg2, const string& arg3, const string& arg4);
RR_DECLSPEC string              Format(const string& str1, const string& arg1, const string& arg2, const string& arg3, const string& arg4, const string& arg5);

RR_DECLSPEC string              Format(const string& str1, const unsigned int& arg1, const string& arg2);
RR_DECLSPEC string              Format(const string& str1, const unsigned int& arg1, const string& arg2, const string& arg3);
RR_DECLSPEC string              Format(const string& str1, const string& arg1, const int& arg2, const string& arg3);
RR_DECLSPEC string              Format(const string& str1, const unsigned int& arg1, const unsigned int& arg2, const string& arg3, const string& arg4);

RR_DECLSPEC string              Append(const string& str);
RR_DECLSPEC string              Append(const int& str);
RR_DECLSPEC string              Append(const unsigned int& str);
RR_DECLSPEC string              Append(const string& s1, const string& s2);
RR_DECLSPEC string              Append(const string& s1, const string& s2, const string& s3);
RR_DECLSPEC string              Append(const string& s1, const unsigned int& s2, const string& s3);
RR_DECLSPEC string              Append(const string& s1, const unsigned int& s2, const string& s3, const string& s4);

RR_DECLSPEC string              Substitute(const string& src, const string& thisOne, const string& withThisOne, const int& howMany = -1);
RR_DECLSPEC string              Substitute(const string& src, const string& thisOne, const int& withThisOne, const int& howMany = -1);
RR_DECLSPEC string              RemoveNewLines(const string& str, const int& howMany = -1);
}
#endif
