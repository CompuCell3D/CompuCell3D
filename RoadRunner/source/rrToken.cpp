#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include "rrToken.h"
//---------------------------------------------------------------------------

namespace rr
{

Token::Token(const CodeTypes& code)
:
tokenCode(code),
tokenDouble(0),
tokenInteger(0),
tokenString(""),
tokenValue(0) // Used to retrieve int or double
{}

}

