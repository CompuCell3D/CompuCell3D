#pragma hdrstop
#include "rrRoadRunner.h"
#include "rrRoadRunnerList.h"
//---------------------------------------------------------------------------


namespace rr
{
RoadRunnerList::RoadRunnerList(const int& nrOfRRs, const string& tempFolder)
{
	string installFolder = "..";
#if defined(_WIN32) || defined(WIN32)
            string compiler(JoinPath(installFolder, "compilers\\tcc\\tcc.exe"));
#elif defined(__linux)
            string compiler("gcc");
#else
            string compiler("gcc");
#endif

    for(int i = 0; i < nrOfRRs; i++)
    {
        RoadRunner* rri = new RoadRunner(tempFolder, JoinPath(installFolder, "rr_support"), compiler);
    	mRRs.push_back(rri);
        mRRs[i]->setTempFileFolder(tempFolder);
    }
}

RoadRunnerList::~RoadRunnerList()
{
    for(int i = 0; i < mRRs.size(); i++)
    {
    	delete mRRs[i];
    }
}

unsigned int RoadRunnerList::count()
{
	return mRRs.size();
}

RoadRunner*	RoadRunnerList::operator[](const int& index)
{
	if(index > -1 && index < count())
    {
		return mRRs[index];
    }
    return NULL;
}
}
