#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <iomanip>
#include "rrStringBuilder.h"
#include "rrStringUtils.h"
#include "rrLogger.h"
//---------------------------------------------------------------------------

using namespace std;
namespace rr
{

StringBuilder::StringBuilder(const string& aStr)
{
    mStringing<<aStr;
}

string StringBuilder::ToString()
{
    return mStringing.str();
}

void StringBuilder::Clear()
{
    mStringing.str("");
}

stringstream& StringBuilder::operator<<(const string& str)
{
    mStringing<<str;
    Log(lDebug5)<<"Appended :"<<RemoveNewLines(str, 1);
    return mStringing;
}

stringstream& StringBuilder::operator<<(const char& ch)
{
    mStringing<<ch;
    Log(lDebug5)<<"Appended :"<<ch;
    return mStringing;
}

void StringBuilder::NewLine(const string& line)
{
    mStringing<<"\n"<<line<<endl;
}

void StringBuilder::Line(const string& line)
{
    mStringing<<line<<endl;
}

void StringBuilder::TLine(const string& line, const int& nrTabs)
{
    string tabs;
    for(int i = 0; i < nrTabs; i++)
    {
        tabs +="\t";
    }

    mStringing<<tabs<<line<<endl;
}

}
