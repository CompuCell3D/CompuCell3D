#ifndef rrRoadRunnerListH
#define rrRoadRunnerListH
#include <vector>
#include "rrObject.h"
//---------------------------------------------------------------------------

namespace rr
{
using std::vector;
class RoadRunner;
class RR_DECLSPEC RoadRunnerList : public rrObject
{
	private:

    protected:
		vector<RoadRunner*>		mRRs;

    public:
								RoadRunnerList(const int& nrOfRRs, const string& tempFolder = gEmptyString);
		virtual				   ~RoadRunnerList();
		RoadRunner*				operator[](const int& index);
        unsigned int			count();

};

}
#endif
