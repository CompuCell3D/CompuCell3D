#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <sstream>
#include "rrStringUtils.h"
#include "rrBaseParameter.h"
#include "rrParameter.h"
//---------------------------------------------------------------------------

namespace rr
{

BaseParameter::BaseParameter(const string& name, const string& hint)
:
mName(name),
mHint(hint)
{}

BaseParameter::~BaseParameter()
{}

string BaseParameter::getName() const
{
    return mName;
}

string BaseParameter::getHint() const
{
    return mHint;
}

string BaseParameter::getValueAsString() const
{
	BaseParameter* ptr = const_cast<BaseParameter*>(this);

	if(dynamic_cast< Parameter<int>* >( ptr))
    {
        return dynamic_cast< Parameter<int>* >(ptr) ->getValueAsString();
    }

    if(dynamic_cast< Parameter<double>* >(ptr))
    {
        return "double";
    }

    if(dynamic_cast< Parameter<bool>* >(ptr))
    {
        return "boolean";
    }

    return "Need to cast to subclass!";
}

string BaseParameter::asString() const
{
    stringstream val;

    val<<"Name: "<<mName<<endl;
    val<<"Type: "<<getType()<<endl;
    val<<"Value: "<<getValueAsString()<<endl;
    val<<"Hint: "<<mHint<<endl;
    return val.str();
}

string BaseParameter::getType() const
{
    string val("no info");

    //Downcasts
    BaseParameter* ptr = const_cast<BaseParameter*>(this);

    if(dynamic_cast< Parameter<int>* >(ptr))
    {
        return "integer";
    }

    if(dynamic_cast< Parameter<double>* >(ptr))
    {
        return "double";
    }

    if(dynamic_cast< Parameter<bool>* >(ptr))
    {
        return "boolean";
    }

    return val;
}

ostream& operator<<(ostream& stream, const BaseParameter& outMe)
{
    stream<<outMe.asString();   //virtual friend idiom
    return stream;
}

//template<>
//string BaseParameterType<double>::getValueAsString() const
//{
//    return ToString(mValue, "%G");
//}
//
//template<>
//string BaseParameterType<int>::getValueAsString() const
//{
//    return ToString(mValue);
//}

}
