#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include "rrRule.h"
//---------------------------------------------------------------------------




namespace rr
{

RRRule::RRRule(const string& rule, const string& ruleTypeStr)
:
mTheRule(rule),
mRuleTypeStr(ruleTypeStr),
mRuleType(rtUnknown)
{
    AssignType();
}

string RRRule::GetLHS()
{
    //Rules have equal signs, or?
    string lhs = mTheRule.substr(0, mTheRule.find('='));
    return lhs;
}

string RRRule::GetRHS()
{
    string rhs = mTheRule.substr(mTheRule.find('=') + 1);
    return rhs;
}

RuleType RRRule::GetType()
{
    return mRuleType;
}

void RRRule::AssignType()
{
    mRuleType = GetRuleTypeFromString(mRuleTypeStr);
}

RuleType GetRuleTypeFromString(const string& str)
{
    if(str == "Algebraic_Rule")
    {
        return rtAlgebraic;
    }
    else if(str == "Assignment_Rule")
    {
        return rtAssignment;
    }
    else if(str == "Rate_Rule")
    {
        return rtRate;
    }
    else
    {
        return rtUnknown;
    }
}

}
