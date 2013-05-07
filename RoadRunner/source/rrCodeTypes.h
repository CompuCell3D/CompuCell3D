#ifndef rrCodeTypesH
#define rrCodeTypesH
#include "rrObject.h"

namespace rr
{
enum CodeTypes
{
    tEmptyToken = 0,
    tEndOfStreamToken,
    tIntToken,
    tDoubleToken,
    tComplexToken,
    tStringToken,
    tWordToken,
    tEolToken,
    tSemiColonToken,
    tCommaToken,
    tEqualsToken,
    tPlusToken,
    tMinusToken,
    tMultToken,
    tDivToken,
    tLParenToken,
    tRParenToken,
    tLBracToken,
    tRBracToken,
    tLCBracToken,
    tRCBracToken,
    tOrToken,
    tAndToken,
    tNotToken,
    tXorToken,
    tTimeWord1,
    tTimeWord2,
    tTimeWord3,
    tColonToken,
    tPowerToken,
    tLessThanToken,
    tLessThanOrEqualToken,
    tMoreThanToken,
    tMoreThanOrEqualToken,
    tNotEqualToken,
    tReversibleArrow,
    tIrreversibleArrow,
    tStartComment,
    tInternalToken,
    tVarToken,
    tVolToken,
    tExternalToken,
    tExtToken,
    tParameterToken,
    tIfToken,
    tDollarToken,
    tUnaryMinusToken,
    tPrintToken,
    tWhileToken,
    tdefnToken,
    tEndToken,
    tTimeStartToken,
    tTimeEndToken,
    tNumPointsToken,
    tSimulateToken,
    tPointToken
};

}//rr namespace


#endif
