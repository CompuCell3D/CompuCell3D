#ifndef rrStringBuilderH
#define rrStringBuilderH
#include <sstream>
#include <string>
#include "rrObject.h"

using std::stringstream;
using std::string;

namespace rr
{

class RR_DECLSPEC StringBuilder : public rrObject
{
    protected:
        stringstream                 mStringing;

    public:
                                    StringBuilder(const string& aStr = "");
        stringstream&          		operator<<(const string& str);
        stringstream&          		operator<<(const char& ch);
        string                      ToString();

        void                        NewLine(const string& line = "");
        void                        Line(const string& line);
        void                        TLine(const string& line, const int& tabs = 1);
        void                        Clear();
};

}

#endif
