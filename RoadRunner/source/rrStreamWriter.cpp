#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include "rrStreamWriter.h"
//---------------------------------------------------------------------------


using namespace std;

namespace rr
{

StreamWriter::StreamWriter(const string& filePath)
:
mFilePath(filePath)
{
    mFileStream.open(filePath.c_str(), ios::trunc);
}

bool StreamWriter::WriteLine(const string& line)
{
    if(mFileStream.is_open())
       {
           mFileStream << line <<std::endl;
           return true;
       }
    return false;
}

bool StreamWriter::Write(const string& text)
{
    if(mFileStream.is_open())
       {
           mFileStream << text;
           return true;
       }
    return false;
}

bool StreamWriter::Close()
{
    mFileStream.close();
    return true;
}

}//namespace rr

