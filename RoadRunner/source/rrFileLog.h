#ifndef rrFileLogH
#define rrFileLogH

//#if defined (__MINGW32__)
//#include <tr1/memory>
//#else
//#include <memory>
//#endif

#include "stdio.h"
#include "rrObject.h"
#include "rrLogLevel.h"
#include "rrLogFile.h"


//#if defined(VS2008)
//#define unique_ptr shared_ptr
//#endif

//using std::unique_ptr;

//Global class holding logfile and other settings. Should Persist trougout the life of the application that is using it. Based on RAII
namespace rr
{

//To init, use
// something like:            gLog.Init("", gLog.GetLogLevel(), unique_ptr<LogFile>(new LogFile("fName.log") )));
class RR_DECLSPEC FileLog : public rrObject
{
    private:
        //unique_ptr<LogFile>     mLogFile;
        LogFile				   *mLogFile;
        string                  mLogPrefix;
        LogLevel                mLogLevel;
        static int              mNrOfInstances;

                                // prevent copying and assignment
                                FileLog(const FileLog& logFile);
        FileLog&                operator = (const FileLog &);

    public:
                                FileLog();
                               ~FileLog();
        string                  GetLogPrefix();
        void                    SetLogPrefix(const string& prefix);
        LogLevel                GetLogLevel();
        void                    SetCutOffLogLevel(const LogLevel& lvl);
//        bool                    Init(const string& logPrefix = "none", const LogLevel& level = lDebug5, unique_ptr<LogFile> logFile = unique_ptr<LogFile>());
        bool                    Init(const string& logPrefix = "none", const LogLevel& level = lDebug5, LogFile* logFile = new LogFile);
        void                    write(const char* str);
        bool                    mLogToServer;
        string                  GetLogFileName();
        int                     GetNrOfInstances();
        FILE*                   GetLogFileHandle();
        string					GetCurrentLogLevel();
};

RR_DECLSPEC extern FileLog gLog;
}

#endif
