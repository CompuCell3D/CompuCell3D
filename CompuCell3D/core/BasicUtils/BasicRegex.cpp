/*******************************************************************\

              Copyright (C) 2004 Joseph Coffland

    This program is free software; you can redistribute it and/or
     modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 2
        of the License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
             GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
     along with this program; if not, write to the Free Software
      Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
                           02111-1307, USA.

            For information regarding this software email
                   jcofflan@users.sourceforge.net

\*******************************************************************/
#include "BasicRegex.h"
#include "BasicException.h"
#include "BasicSmartPointer.h"

#include <string.h>

using namespace std;

BasicRegEx::BasicRegEx(const char *regex) {
  ASSERT_OR_THROW("BasicRegEx() regex cannot be NULL!", regex);
  this->regex = new char[strlen(regex) + 1];
  strcpy(this->regex, regex);

  err = regcomp(&preg, this->regex, REG_EXTENDED);
}

BasicRegEx::~BasicRegEx() {
  delete [] regex;
  regfree(&preg);
}

string BasicRegEx::getErrorStr() {
  if (!err) return "";

  int msgLen = regerror(err, &preg, 0, 0);

  BasicSmartPointer<char> buf = new char[msgLen];
  regerror(err, &preg, buf.get(), msgLen);

  return string(buf.get());
}

bool BasicRegEx::isMatch(const char *s) {
  regmatch_t match;

  if (regexec(&preg, s, 1, &match, 0) != 0)
    return false;

  if (match.rm_so != 0 || match.rm_eo != (int)strlen(s))
    return false;

  return true;
}
