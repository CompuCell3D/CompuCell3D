#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <iostream>
#include <limits>
#include "rrSBMLSymbol.h"
#include "rrStringUtils.h"
#include "rrSBMLSymbolDependencies.h"
//---------------------------------------------------------------------------


using namespace std;
namespace rr
{

SBMLSymbol::SBMLSymbol()
:
mValue(std::numeric_limits<double>::quiet_NaN()),//Represents an un-initialized value
mConcentration(mValue),
mAmount(mValue),
mHasRule(false)
{
//    mDependencies = new SBMLSymbolDependencies;
}

SBMLSymbol::SBMLSymbol(const SBMLSymbol& cp)
:
mId(cp.mId),
mType(cp.mType),
mValue(cp.mValue),
mConcentration(mValue),     //tie reference.. does this work?
mAmount(mValue),            //tie reference.. does this work?
IsSetAmount(cp.IsSetAmount),
IsSetConcentration(cp.IsSetConcentration),
mInitialAssignment(cp.mInitialAssignment),
mHasRule(cp.mHasRule),
mRule(cp.mRule)
{
  *this  = (cp);
}

SBMLSymbol::~SBMLSymbol()
{
//    delete mDependencies;
}

SBMLSymbol& SBMLSymbol::operator =(const SBMLSymbol& rhs)
{
    //Copy properties, one by one
    mId = rhs.mId;
    mType = rhs.mType;
    mValue = rhs.mValue;
    IsSetAmount = rhs.IsSetAmount;
    IsSetConcentration = rhs.IsSetConcentration;
    mInitialAssignment = rhs.mInitialAssignment;
    mHasRule = rhs.mHasRule;
    mRule = rhs.mRule;
    mDependencies = (rhs.mDependencies);
    return *this;
}

int    SBMLSymbol::NumberOfDependencies()
{
    return mDependencies.Count();
}

void SBMLSymbol::AddDependency(SBMLSymbol* symbol)
{
    mDependencies.Add(symbol);
}

SBMLSymbol SBMLSymbol::GetDependency(const int& i)
{
    return mDependencies.At(i);
}

bool SBMLSymbol::HasValue()
{
    return IsNaN(mValue) ? false : true;
}

bool SBMLSymbol::HasInitialAssignment() const
{
    return mInitialAssignment.size() ? true : false;
}

bool SBMLSymbol::HasRule()
{
    return mRule.size() ? true : false;
}


ostream& operator<<(ostream& stream, const SBMLSymbol& symbol)
{
    //stream symbol to stream
    stream<<"ID = "<<            symbol.mId                                <<endl;
    stream<<"Type = "<<         symbol.mType                            <<endl;
    stream<<"Value = "<<        symbol.mValue                            <<endl;
    stream<<"Has Initial Assignment = "<<symbol.HasInitialAssignment()     <<endl;

    if(symbol.HasInitialAssignment())
    {
        stream<<"Intial Assignment = "<<symbol.mInitialAssignment            <<endl;
    }

    stream<<"Has Rule = "<<        ToString(symbol.mHasRule)                <<endl;
    if(symbol.mHasRule)
    {
        stream<<"Rule = "<<        symbol.mRule                    <<endl;
    }
    return stream;
}

}
