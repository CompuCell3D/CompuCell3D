#ifndef rrCapsSupportH
#define rrCapsSupportH
#include <vector>
#include "rrObject.h"
#include "rrCapability.h"
//---------------------------------------------------------------------------

using std::vector;
namespace rr
{

class Capability;
class RoadRunner;

//This is "RoadRunners Capabilities"
class RR_DECLSPEC CapsSupport : public rrObject
{
    protected:
        string                          mName;
        string                          mDescription;
        vector<Capability>     			mCapabilities;
        RoadRunner                     *mRoadRunner;

    public:
                                        CapsSupport(RoadRunner* rr = NULL);
        void                            Add(const Capability& section);
        string                          AsXMLString();
        u_int                           Count();
};

}
#endif


