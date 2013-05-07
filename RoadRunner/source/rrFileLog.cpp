#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <stdexcept>
#include "rrFileLog.h"

//---------------------------------------------------------------------------
namespace rr
{

FileLog gLog;

int FileLog::mNrOfInstances = 0;

FileLog::FileLog()
:
//mLogFile(unique_ptr<LogFile>(new LogFile("Log.txt"))),
mLogFile(new LogFile("Log.txt")),
mLogPrefix("none"),
mLogLevel(lInfo),
mLogToServer(false)
{

    mNrOfInstances++;
}

FileLog::~FileLog()
{
    mNrOfInstances--;
}

int  FileLog::GetNrOfInstances()
{
    return mNrOfInstances;
}

FILE* FileLog::GetLogFileHandle()
{
//    return mLogFile.get()->mFILEHandle;
    return mLogFile->mFILEHandle;
}


//        gLog.Init("", gLog.GetLogLevel(), unique_ptr<LogFile>(new LogFile(logFile.c_str())));
//bool FileLog::Init(const string& logPrefix, const LogLevel& level, unique_ptr<LogFile> logFile)
//{
//    mLogPrefix = logPrefix;
//    mLogLevel = level;
//    mLogFile = move(logFile);
//    return mLogFile.get() ? true : false;
//}

bool FileLog::Init(const string& logPrefix, const LogLevel& level, LogFile* logFile)
{
    mLogPrefix = logPrefix;
    mLogLevel = level;
    mLogFile = logFile;
    return mLogFile != NULL ? true : false;
}

string FileLog::GetLogFileName()
{
    if(mLogFile)
    {
        return mLogFile->GetFileName();
    }
    return string("<none>");
}

LogLevel FileLog::GetLogLevel()
{
    return mLogLevel;
}

string FileLog::GetCurrentLogLevel()
{
	return GetLogLevelAsString(mLogLevel);
}

void FileLog::SetCutOffLogLevel(const LogLevel& lvl)
{
    mLogLevel = lvl;
}

void FileLog::SetLogPrefix(const string& prefix)
{
    mLogPrefix = prefix;
}

string FileLog::GetLogPrefix()
{
    return mLogPrefix;
}

void FileLog::write(const char* str)
{
//    if(!mLogFile.get())
    if(!mLogFile || !mLogFile->mFILEHandle)
    {
        return;
    }
    fprintf(mLogFile->mFILEHandle, "%s", str);

    if (EOF == fflush(mLogFile->mFILEHandle))
    {
        throw std::runtime_error("file write failure");
    }
}

}
