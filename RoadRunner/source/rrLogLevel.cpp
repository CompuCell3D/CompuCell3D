//#ifdef USE_PCH
//#include "rr_pch.h"
//#endif
//#pragma hdrstop
#include <algorithm>
#include "rrLogLevel.h"
//---------------------------------------------------------------------------

using namespace std;

namespace rr
{

int GetHighestLogLevel(){return lAny;}

LogLevel GetLogLevel(const string& lvl)
{
    string level = ToUpperCase(lvl);
    if (level == "ANY")           return lAny;
    if (level == "DEBUG5")        return lDebug5;
    if (level == "DEBUG4")        return lDebug4;
    if (level == "DEBUG3")        return lDebug3;
    if (level == "DEBUG2")        return lDebug2;
    if (level == "DEBUG1")        return lDebug1;
    if (level == "DEBUG")         return lDebug;
    if (level == "INFO")          return lInfo;
    if (level == "WARNING")       return lWarning;
    if (level == "ERROR")         return lError;

    return lAny;
}

string GetLogLevelAsString(const LogLevel& level)
{
    switch (level)
    {
        case lAny   :   return "ANY";
        case lDebug5:   return "DEBUG5";
        case lDebug4:   return "DEBUG4";
        case lDebug3:   return "DEBUG3";
        case lDebug2:   return "DEBUG2";
        case lDebug1:   return "DEBUG1";
        case lDebug :   return "DEBUG";
        case lInfo  :   return "INFO";
        case lWarning:  return "WARNING";
        case lError :   return "ERROR";
        default:
            return "ANY";
    }
}

LogLevel GetLogLevel(const int& level)
{
    switch (level)
    {
        case -1:  return lShowAlways;
        case 0:   return lError;
        case 1:   return lWarning;
        case 2:   return lInfo;
        case 3:   return lDebug;
        case 4:   return lDebug1;
        case 5:   return lDebug2;
        case 6:   return lDebug3;
        case 7:   return lDebug4;
        case 8:   return lDebug5;
        case 9:   return lAny;
        default:
            return lAny;
    }
}

string ToUpperCase(const string& inStr)
{
    string rString(inStr);
    std::transform(rString.begin(), rString.end(), rString.begin(), ::toupper);
    return rString;
}

}

