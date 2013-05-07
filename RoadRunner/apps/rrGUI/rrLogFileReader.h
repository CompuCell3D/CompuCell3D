#ifndef rrLogFileReaderH
#define rrLogFileReaderH
//---------------------------------------------------------------------------
#include <string>
#include <fstream>
#include "mtkThread.h"

class TMForm;
namespace rr
{
using namespace std;
class LogFileReader : public mtkThread
{
    protected:
        ifstream                mFS;
        string                  mFileName;
        TMForm*                 mMainForm;

    public:
                                LogFileReader(const string& fName = "", TMForm* mainForm = NULL);
        void                    Worker();
        void                    SetFileName(const string& fName);
};


}
#endif
