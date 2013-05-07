#ifndef rrCodeBuilderH
#define rrCodeBuilderH
#include "rrStringBuilder.h"

namespace rr
{

class RR_DECLSPEC CodeBuilder : public StringBuilder
{
    protected:
        int                     mSizeOfVarField1;
        int                     mSizeOfVarField2;
        int                     mSizeOfVarField3;
        string                  mDeclSpec;
        string                  mCallingConvention;

    public:
                                CodeBuilder(const string& aStr = "", const string& decl_spec = "D_S", const string& call_conv = "__cdecl");
        void                    FormatVariable(const string& type, const string& varName, const string& comment = "");
        void                    AddFunctionExport(const string& retValue, const string& funcProto);
        void                    AddFunctionProto(const string& retValue, const string& funcProto);
        void                    FormatArray(const string& type, const string& varName, const int& arraySize, const string& comment = "");
};

}

#endif
