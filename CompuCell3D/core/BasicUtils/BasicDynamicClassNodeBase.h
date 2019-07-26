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

#ifndef BASICDYNAMICCLASSNODEBASE_H
#define BASICDYNAMICCLASSNODEBASE_H

#include <cstdlib>
#include <cwchar>

class BasicDynamicClassFactory;


/** 
 * BasicDynamicClassNodeBase is a generic node of a BasicDynamicClass.
 * See BasicDynamicClassFactory.
 *
 * If you don't understand this class don't use it!
 */
class BasicDynamicClassNodeBase {
protected:
  BasicDynamicClassFactory *factory;
  std::size_t offset;

public:
  BasicDynamicClassNodeBase() : factory(0), offset(0) {}

  /** 
   * See BasicDynamicClassNode::getSize()
   */
  virtual size_t getSize() const = 0;

  /** 
   * See BasicDynamicClassNode::_init()
   */
  virtual void _init(void *x) = 0;

  /** 
   * This function should not be used directly because it is not type safe.
   * 
   * @param x The dynamic class.
   * 
   * @return A pointer to the memory allocated for this class node.
   */
  void *getNode(const void *x) const;

  /** 
   * Register this node with the factory.
   * If this node has already been registered with some factory a 
   * BasicException will be thrown.
   * 
   * @param factory The factory with which to register.
   */  
  void registerNode(BasicDynamicClassFactory *factory);

protected:
  /** 
   * Called by BasicDynamicClassFactory to set the offset in the dynamic 
   * class to this class node.
   * 
   * @param offset The byte offset.
   */  
  void setOffset(const size_t offset) {this->offset = offset;}

  virtual ~BasicDynamicClassNodeBase() {}

  friend class BasicDynamicClassFactory;
};

#endif
