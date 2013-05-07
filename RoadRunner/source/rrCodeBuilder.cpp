#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <iomanip>
#include "rrStringUtils.h"
#include "rrCodeBuilder.h"
//---------------------------------------------------------------------------
using namespace std;
namespace rr
{

CodeBuilder::CodeBuilder(const string& aStr, const string& decl_spec, const string& call_conv)
:
mSizeOfVarField1(45),
mSizeOfVarField2(55),
mSizeOfVarField3(45),
mDeclSpec(decl_spec),
mCallingConvention(call_conv)
{
    mStringing<<aStr;
}

void CodeBuilder::FormatVariable(const string& type, const string& varName, const string& comment)
{

    mStringing<<left<<setw(mSizeOfVarField1)<<type    <<varName<<     setw(mSizeOfVarField2)<<";";
    if(comment.size())
    {
        mStringing<<"//"<<comment;
    }

    mStringing<<endl;
}

void CodeBuilder::AddFunctionExport(const string& retValue, const string& funcProto)
{
    //mStringing<<mDeclSpec<<" "<<left<<setw(mSizeOfVarField1)<<retValue<<mCallingConvention<<setw(mSizeOfVarField2)<<funcProto + ";"<<endl;
    mStringing<<mDeclSpec<<" "<<left<<setw(mSizeOfVarField1)<<retValue<<setw(mSizeOfVarField2)<<funcProto + ";"<<endl;
}

void CodeBuilder::AddFunctionProto(const string& retValue, const string& funcProto)
{
    mStringing<<"   "<<" "<<left<<setw(mSizeOfVarField1)<<retValue<<setw(mSizeOfVarField2)<<funcProto + ";"<<endl;
}

void CodeBuilder::FormatArray(const string& type, const string& varName, const int& _arraySize, const string& comment)
{
    int arraySize = _arraySize;
    if(arraySize == 0)
    {
        //an array of zero length is undefined.. don't put it in the header..
        mStringing<<"\n//The array size for the follwoing variable was generated as 0. We put 1, to make it legal code.\n";
         arraySize = 1;
    }

    string field2(varName +"["+ rr::ToString(arraySize)+"];");
    mStringing<<left<<setw(mSizeOfVarField1)<<type    << setw(mSizeOfVarField2)<<field2;

    if(comment.size())
    {
        mStringing<<left<<setw(mSizeOfVarField3)<<"//" + comment;
    }
    mStringing<<"\n";

    //Add the size for each array, so we don't have to calculate later on..
    if(_arraySize == 0)
    {
        arraySize = 0;
    }

    mStringing<<left<<setw(mSizeOfVarField1)<<"D_S const int"    << setw(mSizeOfVarField2)<<varName + "Size=" + rr::ToString(arraySize) + ";";
    mStringing<<endl;
}

}
