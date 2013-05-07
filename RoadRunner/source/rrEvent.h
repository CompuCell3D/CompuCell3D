#ifndef rrEventH
#define rrEventH
#include "rrObject.h"
#include "rrRandom.h"
//---------------------------------------------------------------------------
namespace rr
{


class RR_DECLSPEC Event : public rrObject
{
	protected:
        int			           	mID;
		double		           	mPriority;
        double		           	mDelay;
        Random					mRandom;	//If we need randomness..

    public:
    				           	Event(const int& id, const double& prior = 0, const double& delay = 0);
    				           	Event(const Event& id);
    				           ~Event(){}
		double		           	GetPriority() const;
        void					SetPriority(const double& prior);
		int		           		GetID() const;
		Event&					operator=(const Event& rhs);
		friend bool 			operator<(const Event& e1, const Event& e2);
		friend bool 			operator==(const Event& e1, const Event& e2);
        friend ostream&         operator<<(ostream& str, const Event& event);
};

struct RR_DECLSPEC SortByPriority
{
	bool operator()( const rr::Event& lx, const rr::Event& rx ) const
    {
        return lx.GetPriority() > rx.GetPriority();
    }
};

};
#endif
