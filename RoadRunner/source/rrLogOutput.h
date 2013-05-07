#ifndef rrLogOutputH
#define rrLogOutputH
#include <sstream>
#include <string>
#include <stdio.h>
#include "rrObject.h"
#include "rrLogLevel.h"
using std::string;
using std::ostringstream;

namespace rr
{

class RR_DECLSPEC LogOutput : public rrObject
{
    public:
                                LogOutput();
        static bool             mShowLogTime;
        static bool             mShowLogPrefix;
        static bool             mShowLogLevel;
        static bool             mUseLogTabs;
        static bool             mLogToConsole;
        static bool             mDoLogging;
        static void             Output(const string& msg, const LogLevel& lvl);
        static void             StopLogging();
        static void             StartLogging();
};

}
#endif



