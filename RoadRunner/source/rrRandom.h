#ifndef rrRandomH
#define rrRandomH
//---------------------------------------------------------------------------
#include "rrObject.h"
#include "mtrand/mtrand.h"

namespace rr
{

class RR_DECLSPEC Random :rrObject
{

    private:
		MTRand			mRand;		//Get a double in [0, 1)
    public:
    					Random();
        double          NextDouble() const;
};

}
#endif
