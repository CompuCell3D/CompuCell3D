#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include "rrEvent.h"
#include "rrRandom.h"
//---------------------------------------------------------------------------

namespace rr
{

Event::Event(const int& id, const double& prior, const double& delay)
:
mID(id),
mPriority(prior),
mDelay(delay)
{}

Event::Event(const Event& rhs)
{
    (*this) = rhs;
}

Event& Event::operator=(const Event& rhs)
{
	if(this != &rhs)
    {
    	(*this).mID 		= rhs.mID;
    	(*this).mPriority 	= rhs.mPriority;
    	(*this).mDelay 		= rhs.mDelay;
    }

	return *this;
}

int	Event::GetID() const
{
	return mID;
}

void Event::SetPriority(const double& prior)
{
	mPriority = prior;
}

double Event::GetPriority() const
{
	return mPriority;
}

//Friend functions
bool operator==(const Event &e1, const Event &e2)
{
	if(e1.mID == e2.mID && e1.mPriority == e2.mPriority && e1.mDelay == e2.mDelay)
    {
    	return true;
    }
  	return false;
}

bool operator<(const Event &e1, const Event &e2)
{
 	if(e1.mPriority == e2.mPriority && e1.mPriority !=0 && e1.mID != e2.mID)
    {
		//Random toss...
		return (e1.mRandom.NextDouble() > 0.5) ? false : true;
    }

	return e1.mPriority >= e2.mPriority;	//Used in sorting algorithm
}

ostream& operator<<(ostream& stream, const Event& anEvent)
{
	stream<<string("Event ID: ");
    stream<<anEvent.mID;
    stream<<string(" Priority: ");
    stream<<anEvent.mPriority;
	return stream;
}

}//namespace
