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

#ifndef BASICMODULEINFO_H
#define BASICMODULEINFO_H

#include <iostream>
#include <string>
#include <string>
#include <stdlib.h>

class BasicModuleInfo{


public:
    BasicModuleInfo():versionMajor(0),versionMinor(0),versionSubMinor(0)
    {}

    
  /// Plugin name.
  std::string name;

  std::string type;

  std::string fileName;

  /// Plugin description.
  std::string description;

  std::string author;

 int versionMajor;
 int versionMinor;
 int versionSubMinor;



  friend std::ostream &operator<<(std::ostream &, BasicModuleInfo &);
};

std::ostream &operator<<(std::ostream &stream, BasicModuleInfo &info) {
    stream <<std::endl;
    stream <<"****** "<<info.name<<" - "<<info.type<<" by "<<info.author<<std::endl;
    stream <<"Version: "<<info.versionMajor<<"."<<info.versionMinor<<"."<<info.versionSubMinor<<std::endl; 
    stream << "File: "<<info.fileName<<std::endl;
    stream << std::endl;
  return stream;
}

#endif
