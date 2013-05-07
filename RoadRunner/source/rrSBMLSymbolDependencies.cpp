#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <iostream>
#include <limits>
#include "rrSBMLSymbolDependencies.h"
#include "rrStringUtils.h"
#include "rrSBMLSymbol.h"
//---------------------------------------------------------------------------
using namespace std;
namespace rr
{

SBMLSymbolDependencies::SBMLSymbolDependencies(const SBMLSymbolDependencies& cp)
{
    mDependencies = cp.mDependencies;
}

SBMLSymbolDependencies::~SBMLSymbolDependencies()
{

}

SBMLSymbolDependencies& SBMLSymbolDependencies::operator=(const SBMLSymbolDependencies& rhs)
{
    mDependencies = rhs.mDependencies;
    return *this;
}

void SBMLSymbolDependencies::Add(SBMLSymbol* symbol)
{
    SBMLSymbol *symbDep = new SBMLSymbol( (*symbol) );

    mDependencies.push_back(symbDep);    //Makes a copy
}

int SBMLSymbolDependencies::Count()
{
    return mDependencies.size();
}

SBMLSymbol    SBMLSymbolDependencies::At(const int& i)
{
    return *mDependencies[i];
}
}
