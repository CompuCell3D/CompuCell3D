#ifndef rrLogFileH
#define rrLogFileH
#include <memory>
#include <string>
#include <fstream>
#include "rrLogLevel.h"
#include "rrObject.h"

using std::FILE;
using std::string;
//Global class holding logfile and other settings. Persist trougout the life of the application that is using it. Based on RAII

namespace rr
{
class RR_DECLSPEC LogFile : public rrObject
{
    private:
                                // prevent copying and assignment
                                LogFile(const LogFile& logFile);
                                LogFile& operator=(const LogFile&);
        string                  mFileName;

    public:
                                LogFile(const string& fName="Log.txt");
                               ~LogFile();
        FILE*                   mFILEHandle;
        string                  GetFileName();
};

}
#endif
