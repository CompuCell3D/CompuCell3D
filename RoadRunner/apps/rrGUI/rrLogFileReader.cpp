#ifdef MTK_PCH
#include "mtk_pch.h"
#endif
#pragma hdrstop
#include <Classes.hpp>
#include "rrLogFileReader.h"
#include "rrLogger.h"
#include "MainForm.h"
//---------------------------------------------------------------------------

namespace rr
{
LogFileReader::LogFileReader(const string& file, TMForm* mainForm)
:
mFileName(file),
mMainForm(mainForm)
{
    if(FileExists(mFileName.c_str()))
    {
        mFS.open(file.c_str());
    }
}

void LogFileReader::SetFileName(const string& fName)
{
    mFileName = fName;
}

void LogFileReader::Worker()
{
    mIsRunning = true;
	mIsFinished = false;

    //First advance to end
    if(!mFS.is_open())
    {
        mFS.open(mFileName.c_str());
        if(!mFS)
        {
            mIsTimeToDie = true;
        }
    }
    mFS.seekg (0, ios::end);
    streampos pos = mFS.tellg();
    streampos lastPosition = pos;

	while(!mIsTimeToDie)
	{
        //Read until end of file
        while(!mFS.eof())
        {
            char* data = new char[2048];
            mFS.getline(data, 2048);

            if(strlen(data) > 1)
            {
                if(mMainForm)
                {
                    while(mMainForm->mLogString)
                    {
                        ;
                    }
                    mMainForm->mLogString = new string(data);
                    TThread::Synchronize(NULL, mMainForm->LogMessage);
                }
            }
            delete [] data;

        }
        if(mFS.fail())
        {
            mFS.clear();
        }
        pos = mFS.tellg();
        Sleep(100);
	}

    mIsRunning = false;
	mIsFinished = true;
}

}

