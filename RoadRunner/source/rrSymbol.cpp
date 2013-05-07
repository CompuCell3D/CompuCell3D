#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include "rrSymbol.h"
//---------------------------------------------------------------------------
namespace rr
{

Symbol::Symbol(const string& _name, const double& _value)
:
name(_name),
value(_value),
rateRule(false),
hasOnlySubstance(false)
{}

Symbol::Symbol(const string& _keyName, const string& _name, const double& _value)
:
keyName(_keyName),
name(_name),
value(_value),
hasOnlySubstance(false),
rateRule(false)
{}

Symbol::Symbol(const string& _name, const double& _value, const string& _compartmentName)
:
name(_name),
value(_value),
compartmentName(_compartmentName),
hasOnlySubstance(false),
rateRule(false)
{}

Symbol::Symbol(const string& _name, const double& _value, const string& _compartmentName, const string& _formula)
:
name(_name),
value(_value),
compartmentName(_compartmentName),
formula(_formula),
hasOnlySubstance(false),
rateRule(false)
{}

ostream& operator<<(ostream& stream, const Symbol& symbol)
{
    stream<<"Name: "<<symbol.name;
    return stream;
}
}
