#ifndef rrLogContainerH
#define rrLogContainerH
//---------------------------------------------------------------------------
#include <sstream>
#include <string>
#include <stdio.h>
#include "rrObject.h"
#include "rrLogLevel.h"

using std::string;
using std::ostringstream;

namespace rr
{

template <class T>
class RR_DECLSPEC LogContainer : public rrObject
{
    private:
        LogLevel                    mCurrentLogLevel;
                                    LogContainer(const LogContainer&);    //Don't copy this one..
    protected:
        std::ostringstream          mOutputStream;

    public:
                                    LogContainer();
        virtual                    ~LogContainer();
        std::ostringstream&         Get(const LogLevel& level);
        string						GetCurrentLogLevel();
};



}
#endif
