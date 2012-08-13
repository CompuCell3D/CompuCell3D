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

#ifndef BASICCLASSGROUPARRAY_H
#define BASICCLASSGROUPARRAY_H

#include "BasicClassGroupFactory.h"
#include "BasicArray.h"

class BasicClassGroup;

/** 
 * A BasicArray and a BasicClassGroupFactory wrapped together.  Allocated
 * BasicClassGroups are kept in an array.  They can then be accessed or
 * deleted using an id which is simply an array index.
 */
class BasicClassGroupArray {
  BasicArray<BasicClassGroup *> groups;
  BasicClassGroupFactory factory;

public:

  /** 
   * Add a new BasicClassGroup to the end of the array.
   * 
   * @return The id or index of the new class group.
   */  
  unsigned int create() {
    return groups.put(factory.create());
  }

  /** 
   * Destroy an instance of a class group.  If the id is invalid
   * unexpected behavior may occur.
   * 
   * @param id The class group id.
   */
  void destroy(const unsigned int id) {
    factory.destroy(get(id));
  }

  /** 
   * Register a class with the group factory.
   * See BasicClassGroupFactory::registerClass().
   * 
   * @param accessor The class accessor.
   */
  void registerClass(BasicClassAccessorBase *accessor) {
    factory.registerClass(accessor);
  }

  /** 
   * If the id is invalid unexpected behavior may occur.
   * See BasicClassGroupAccessor for information on how to access a class
   * with in the group.
   *
   * @param id A valid class group id.
   * 
   * @return A pointer to the class group.
   */
  BasicClassGroup *get(const unsigned int id) const {return groups[id];}

  /** 
   * See BasicClassGroupArray::get()
   */
  BasicClassGroup *operator[](const unsigned int id) const {return groups[id];}

  /** 
   * See BasicArray::getSize()
   *
   * @return The current size of the array.
   */
  unsigned int getSize() const {return groups.getSize();}
};

#endif
