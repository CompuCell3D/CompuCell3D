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
#include "BasicStringTable.h"
#include "BasicException.h"

#include <string.h>

BasicStringTable:: ~BasicStringTable() {
  table_t::iterator it;
  for (it = table.begin(); it != table.end(); it++)
    if (*it) delete *it;
}

const char *BasicStringTable:: get(const char *str) {
  ASSERT_OR_THROW(str, "BasicStringTable::get() NULL string!");

  table_t::iterator it = table.find(str);
  if (it != table.end()) return *it;

  // Create new entry
  size_t len = strlen(str) + 1;
  char *newStr = new char[len];
  memcpy(newStr, str, len);

  table.insert(newStr);

  return newStr;
}
