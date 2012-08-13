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

#ifndef BASICCLASSFACTORY_H
#define BASICCLASSFACTORY_H

#include "BasicClassFactoryBase.h"


/** 
 * A templated class factory. B is the base class and T the derived class.
 * The base class may be void, but otherwise B must be a base class of T.
 */
template <class B, class T>
class BasicClassFactory: public BasicClassFactoryBase<B> {
public:
  /** 
   * @return A pointer to a newly allocated instance of class T.
   */
  virtual B *create() {return new T;}

  /** 
   * @param classNode A pointer to the instance of class T to deallocate.
   */
  virtual void destroy(B *classNode) {
     delete (B*)classNode;
  }
};
#endif
