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

#ifndef BASICDYNAMICCLASSNODE_H
#define BASICDYNAMICCLASSNODE_H

#include "BasicDynamicClassNodeBase.h"

/** 
 * BasicDynamicClassNode is a node of a BasicDynamicClass.
 * See BasicDynamicClassFactory.
 *
 * If you don't understand this class don't use it!
 */
template <class T>
class BasicDynamicClassNode: public BasicDynamicClassNodeBase {
public:

  /** 
   * @return The size in bytes of the class which this node represents.
   */
  virtual unsigned int getSize() const {return (unsigned int)sizeof(T);}

  /** 
   * @param x A pointer to an instance of a dynamic class allocated by 
   *          BasicDynamicClassFactory.
   * 
   * @return A pointer to the instanceof of the class this node represents.
   */
  T *get(const void *x) const {
    return (T *)getNode(x);
  }

  /** 
   * Calls the specific init function for this class T.
   * 
   * @param x A pointer to an instance of a dynamic class allocated by
   *          BasicDynamicClassFactory.
   */
  virtual void _init(void *x) {init((T *)x);}

  /** 
   * An initialization function which can be overriden in child classes to
   * initialized a class after it is allocated.  This is necessary because
   * the constructors for the class nodes with in a dynamic class are never
   * called.
   * 
   * @param x A pointer to the uninitialized class.
   */
  virtual void init(T *x) {}
};
#endif

