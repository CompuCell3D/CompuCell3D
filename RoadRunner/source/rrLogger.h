#ifndef rrLoggerH
#define rrLoggerH
#include <sstream>
#include <string>
#include <stdio.h>
#include "rrExporter.h"
#include "rrLogContainer.h"
#include "rrLoggerUtils.h"
#include "rrLogLevel.h"
#include "rrFileLog.h"
#include "rrLogOutput.h"

//using std::string;
//using std::ostringstream;

namespace rr
{

class RR_DECLSPEC Logger : public LogContainer<LogOutput>
{};

#ifndef NO_LOGGER
#define Log(level) \
    if (level > rr::GetHighestLogLevel()) { ; }\
    else if (level > gLog.GetLogLevel()) { ; } \
    else Logger().Get(level)
#else
#define Log(level) \
    if (true) {  }\
    else \
    Logger().Get(level)
#endif

}//Namespace rr

#endif
