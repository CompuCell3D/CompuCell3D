#ifndef rrTokenH
#define rrTokenH
#include "rrObject.h"
#include "rrCodeTypes.h"

namespace rr
{
// This is currently used to store the previous token and support simple look ahead
class RR_DECLSPEC Token : public rrObject
{
    protected:

    public:
        CodeTypes         tokenCode;
        double             tokenDouble;
        int             tokenInteger;
        string             tokenString;
        double             tokenValue; // Used to retrieve int or double
                        Token(const CodeTypes& code = tEmptyToken);
};

}

#endif
