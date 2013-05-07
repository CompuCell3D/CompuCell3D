#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <stdio.h>
#include "rrException.h"
#include "rrLogFile.h"
//---------------------------------------------------------------------------

namespace rr
{

using std::fstream;
LogFile::LogFile(const string& name)
:
mFileName(name),
mFILEHandle(fopen(name.c_str(), "w"))
{
    if (!mFILEHandle && name.size())
    {
        //throw Exception("Failed opening log file");
    }
}

LogFile::~LogFile()
{
	if(mFILEHandle)
	{
		fclose(mFILEHandle);
	}
}

string LogFile::GetFileName()
{
	return mFileName;
}
}
