#ifndef rrLogLevelH
#define rrLogLevelH
#include <string>
#include "rrExporter.h"

using std::string;
namespace rr
{

enum  LogLevel
    {
        lShowAlways = -1,
        lError      = 0,
        lWarning    = 1,
        lInfo       = 2,
        lDebug      = 3,
        lDebug1     = 4,
        lDebug2     = 5,
        lDebug3     = 6,
        lDebug4     = 7,
        lDebug5     = 8,
        lAny        = 9,
        lUser
    };

RR_DECLSPEC string   ToUpperCase(const string& inStr);
RR_DECLSPEC int      GetHighestLogLevel();
RR_DECLSPEC LogLevel GetLogLevel(const string& level);
RR_DECLSPEC string   GetLogLevelAsString(const LogLevel& level);
RR_DECLSPEC LogLevel GetLogLevel(const int& lvl);

}
#endif
