#ifndef memoLoggerH
#define memoLoggerH
#include <Vcl.StdCtrls.hpp>
#include <sstream>

//---------------------------------------------------------------------------
//Minimalistic logger to a memo...
class MemoLogger
{
    protected:
        std::ostringstream          mStream;
		TMemo					   *mMemo;
    public:
                                    MemoLogger(TMemo* aMemo);
        virtual                    ~MemoLogger();
        std::ostringstream&         Get();
};

#define Log() \
    MemoLogger(infoMemo).Get()

#endif
