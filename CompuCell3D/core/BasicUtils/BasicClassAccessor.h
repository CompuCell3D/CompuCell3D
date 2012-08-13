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

#ifndef BASICCLASSACCESSOR_H
#define BASICCLASSACCESSOR_H

#include "BasicClassAccessorBase.h"
#include "BasicClassFactory.h"

template <class T>
class BasicClassFactoryBase;

/** 
 * Used to access the classes of a BasicClassGroup in a type safe manner.
 *
 * By registering an instance of this class with BasicClassGroupFactory you
 * can dynamically add classes to a class group.
 */
template <class T>
class BasicClassAccessor: public BasicClassAccessorBase {
public:
  /** 
   * @param group An instance of a class group allocated by 
   *              BasicClassGroupFactory.
   * 
   * @return A pointer the class T with in the given class group.
   */
  T *get(BasicClassGroup *group) const {return (T *)getClass(group);}

protected:
  /** 
   * Used by BasicClassGroupFactory to create a new BasicClassFactory for this
   * class type.
   */
  virtual BasicClassFactoryBase<void> *createClassFactory() {
    return new BasicClassFactory<void, T>;
  }

  virtual void deallocateClass(BasicClassGroup *group) const{
	T * ptr=get(group);
	if (ptr)
		delete ptr;
  }
};

#endif
