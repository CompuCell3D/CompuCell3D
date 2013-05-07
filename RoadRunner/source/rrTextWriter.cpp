#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include "rrTextWriter.h"
//---------------------------------------------------------------------------


using namespace std;
namespace rr
{


TextWriter::TextWriter(ostream& aStream)
:
mStream(aStream)
{

}

void TextWriter::Write(const string& chars)
{
    mStream<<chars;
}

void TextWriter::WriteLine()
{
    mStream<<endl;
}

}

