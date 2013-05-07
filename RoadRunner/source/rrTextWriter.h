#ifndef rrTextWriterH
#define rrTextWriterH
#include <iostream>
#include <string>
#include "rrObject.h"

//---------------------------------------------------------------------------
using std::ostream;
namespace rr
{

class RR_DECLSPEC TextWriter : public rrObject
{
    protected:
        ostream&        mStream;
    public:
                        TextWriter(ostream& aStream);
        void            Write(const string& chars);
        void            WriteLine();
};

}


#endif
