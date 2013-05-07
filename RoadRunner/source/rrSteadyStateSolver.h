#ifndef rrSteadyStateSolverH
#define rrSteadyStateSolverH
#include "rrObject.h"
//---------------------------------------------------------------------------

namespace rr
{

class RR_DECLSPEC ISteadyStateSolver : public rrObject
{
    public:
        virtual double solve(const vector<double>& yin) = 0;
};

}
#endif
