#ifndef rrCapabilityH
#define rrCapabilityH
#include <string>
#include <vector>
#include "rrObject.h"
#include "rrBaseParameter.h"
//---------------------------------------------------------------------------

namespace rr
{
//Create a class later on - Parameters;
typedef vector<rr::BaseParameter*> Parameters;

//Will be renamed to Capability
class RR_DECLSPEC Capability : public rrObject
{
    protected:
        string                              mName;
        string                              mDescription;
        string                              mMethod;
        Parameters				            mParameters;

    public:
                                            Capability(const string& name, const string& method, const string& descr);
                                            Capability(const Capability& fromMe);
		void							 	setup(const string& name, const string& method, const string& descr);
        void                                add(rr::BaseParameter* me);
        string                              asString() const;
        u_int                               nrOfParameters() const;
        const rr::BaseParameter&            operator[](const int& i) const;
        string                              getName() const;
        string                              getDescription() const;
        string                              getMethod() const;
        Parameters*							getParameters();
        rr::BaseParameter*	   				getParameter(const string& paraName);

		RR_DECLSPEC friend ostream&   		operator <<(ostream& os, const Capability& caps);
};

}
#endif
