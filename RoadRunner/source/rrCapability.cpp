#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <sstream>
#include "rrParameter.h"
#include "rrCapability.h"
//---------------------------------------------------------------------------

namespace rr
{

Capability::Capability(const string& name, const string& method, const string& description)
:
mName(name),
mMethod(method),
mDescription(description)
{}

Capability::Capability(const Capability& from)
:
mName(from.mName),
mMethod(from.mMethod),
mDescription(from.mDescription),
mParameters(from.mParameters)
{}

void Capability::setup(const string& name, const string& method, const string& descr)
{
    mName = name;
    mMethod = method;
    mDescription = descr;
}

Parameters* Capability::getParameters()
{
	return &mParameters;
}

const rr::BaseParameter& Capability::operator[](const int& i) const
{
    return *(mParameters[i]);
}

string Capability::getName() const
{
    return mName;
}

string Capability::getDescription() const
{
    return mDescription;
}

string Capability::getMethod() const
{
    return mMethod;
}

u_int Capability::nrOfParameters() const
{
    return mParameters.size();
}

void Capability::add(rr::BaseParameter* me)
{
    mParameters.push_back(me);
}

string Capability::asString()  const
{
    stringstream caps;
    caps<<"Section: " << mName <<endl;
    caps<<"Method: " << mMethod<<endl;
    caps<<"Description: " << mDescription<<endl;

    for(int i = 0; i < nrOfParameters(); i++)
    {
        caps <<*(mParameters[i])<<endl;
    }
    return caps.str();
}


BaseParameter* Capability::getParameter(const string& paraName)
{
	for(int i = 0; i < mParameters.size(); i++)
    {
		if(mParameters[i] && mParameters[i]->mName == paraName)
        {
        	return mParameters[i];
        }
    }
	return NULL;
}

ostream& operator <<(ostream& os, const Capability& caps)
{
	os<<"Capabilities for "<<caps.mName<<"\n";

	for(int i = 0; i < caps.nrOfParameters(); i++)
    {
    	os<< *(caps.mParameters[i]);
    }
	return os;
}
}