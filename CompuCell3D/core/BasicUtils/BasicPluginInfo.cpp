/*******************************************************************\

              Copyright (C) 2003 Joseph Coffland

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

#include "BasicPluginInfo.h"

using namespace std;

ostream &operator<<(ostream &stream, BasicPluginInfo &info) {
  stream << info.name;
  if (info.getNumDeps()) {
    stream << " (";

    bool first = true;
    for (unsigned int i = 0; i < info.getNumDeps(); i++) {
      if (first) first = false;
      else stream << ", ";
      
      stream << info.getDependency(i);
    }

    stream << ")";

  }

  stream << ": " << info.description;

  return stream;
}
