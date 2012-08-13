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

#ifndef BASICPLUGININFO_H
#define BASICPLUGININFO_H

#include <iostream>
#include <string>
#include <string.h>
#include <stdlib.h>

class BasicPluginInfo {
  /// Plugin name.
  std::string name;

  /// Plugin description.
  std::string description;

  /// The number of dependencies in the array
  unsigned int numDeps;

  /// An array of plugin dependencies.  
  char **dependencies;

public:
  BasicPluginInfo(std::string name, std::string description) :
    name(name), description(description), numDeps(0), dependencies(0) {}

  /** 
   * @param name Plugin name.
   * @param description Plugin description.
   * @param numDeps Number of dependencies in the array.
   * @param deps A constant array of dependency names.
   */
  BasicPluginInfo(std::string name, std::string description,
		  const unsigned int numDeps, const char *deps[]) :
    name(name), description(description), numDeps(numDeps) {
    dependencies = new char *[numDeps];

    for (unsigned int i = 0; i < numDeps; i++)
      dependencies[i] = strdup(deps[i]);
  }

  ~BasicPluginInfo() {
    if (dependencies) {
      for (unsigned int i = 0; i < numDeps; i++)
			free(dependencies[i]);

      delete [] dependencies;
    }
  }

  /// Copy constructor
  BasicPluginInfo(const BasicPluginInfo &info) :
    name(info.name), description(info.description), numDeps(info.numDeps),
    dependencies(info.dependencies) {}

  const std::string &getName() const {return name;}
  const std::string &getDescription() const {return description;}
  const unsigned int getNumDeps() const {return numDeps;}
  const std::string getDependency(const int i) const {return dependencies[i];}

  friend std::ostream &operator<<(std::ostream &, BasicPluginInfo &);
};

std::ostream &operator<<(std::ostream &, BasicPluginInfo &);
#endif
