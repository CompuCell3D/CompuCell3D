#pragma hdrstop
#include <vector>
#include "memoLogger.h"
#include "rrUtils.h"
//---------------------------------------------------------------------------
#pragma package(smart_init)

using namespace std;
//Minimalistic logger to a memo...
MemoLogger::MemoLogger(TMemo* aMemo)
:
mMemo(aMemo)
{}

MemoLogger::~MemoLogger()
{
	vector<string> lines = rr::SplitString(mStream.str(),"\n");

    for(int i = 0; i < lines.size(); i++)
    {
	    mMemo->Lines->Add(lines[i].c_str());

    }
}

std::ostringstream& MemoLogger::Get()
{
	return mStream;
}

