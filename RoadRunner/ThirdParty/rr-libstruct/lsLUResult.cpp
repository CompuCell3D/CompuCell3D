#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include "lsLUResult.h"

namespace ls
{

LU_Result::LU_Result()
:
L(NULL),
U(NULL),
P(NULL),
Q(NULL)
{}


LU_Result::~LU_Result()
{
    delete L;
    delete U;
    delete P;
    delete Q;
}

}
