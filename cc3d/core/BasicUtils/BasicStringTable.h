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
#ifndef BASICSTRINGTABLE_H
#define BASICSTRINGTABLE_H

#include <set>

#ifndef LTSTR
#define LTSTR
#include <string.h>

struct ltstr {
  bool operator()(const char* s1, const char* s2) const {
    return strcmp(s1, s2) < 0;
  }
};
#endif // LTSTR


class BasicStringTable {
  typedef std::set<const char *, ltstr> table_t;
  table_t table;

public:
  ~BasicStringTable();

  const char *get(const char *str);
};
#endif // BASICSTRINGTABLE_H
