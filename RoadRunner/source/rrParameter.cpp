#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include "rrStringUtils.h"
#include "rrParameter.h"
//---------------------------------------------------------------------------

namespace rr
{

template<>
string rr::Parameter<double>::getValueAsString() const
{
    return ToString(mValue, "%G");
}

template<>
string rr::Parameter<int>::getValueAsString() const
{
    return ToString(mValue);
}

}
