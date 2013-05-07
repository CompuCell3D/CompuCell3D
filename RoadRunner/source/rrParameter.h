#ifndef rrParameterH
#define rrParameterH
#include "rrObject.h"
#include "rrBaseParameter.h"
//---------------------------------------------------------------------------
namespace rr
{

template<class T>
class Parameter : public BaseParameter
{
    protected:
        T                                   mValue;

    public:
                                            Parameter(const string& name, const T& value, const string& hint);
		void								setValue(const T& val){mValue = val;}
        T									getValue();
        string                      		getValueAsString() const;
};

template<class T>
Parameter<T>::Parameter(const string& name, const T& value, const string& hint)
:
rr::BaseParameter(name, hint),
mValue(value)
{}

template<class T>
string Parameter<T>::getValueAsString() const
{
    return ToString(mValue);
}

template<class T>
T Parameter<T>::getValue()
{
    return mValue;
}

#if defined(_MSC_VER)
template<>
string Parameter<int>::getValueAsString() const;

template<>
string Parameter<double>::getValueAsString() const;
#endif

}
#endif
