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

#ifndef BASICDYNAMICCLASSFACTORY_H
#define BASICDYNAMICCLASSFACTORY_H

#include "BasicArray.h"

class BasicDynamicClassNodeBase;

/**
 * Manages dynamic class aggregation.  BasicDynamicClassNode(s) can be be
 * registered with an instance of a BasicDynamicClassFactory to create
 * type safe dynamically aggregated classes.  
 *
 * Pros:
 * <ul>
 *  <li>Less memory usage, because the classes are all allocated as one 
 *      block of memory.</li>
 *  <li>Data and functionality can be added on the fly allowing pluggins to 
 *      augment classes after compile time.</li>
 * </ul>
 *
 * Cons:
 * <ul>
 *  <li>Constructors and destructors are never called.  This means classes 
 *      like std::string can not be used as class nodes.  Classes can be 
 *      initialized by overriding BasicDynamicClassNode::init().</li>
 *  <li>
 *   Classes with virtual members cannot be reliably used as class nodes.  
 *   Virtual classes have an internal pointer to their class virtual table.  
 *   Unless you can copy this internal pointer virtual classes with not work 
 *   and may yield unexpected behavior.  Assure that there are no virtual 
 *   functions anywhere in the inheritance tree before registering a class as 
 *   a dynamic class node.</li>
 * </ul>
 *
 * Don't use this class unless you know why and how to use it.
 *
 * See also this similar class BasicClassGroupFactory.
 */
class BasicDynamicClassFactory {
  unsigned int classSize;
  unsigned int numClasses;

  BasicArray<BasicDynamicClassNodeBase *> nodes;

public:
  BasicDynamicClassFactory();

  /** 
   * This function will allocate one contiguous block of memory the size of the
   * sum of the sizes of the registered class nodes.  If the registered class 
   * nodes have overridden the BasicDynamicClassNode::init() function it will 
   * be called first.
   * WARNING the constructors of the dynamic class nodes are NOT called.
   * 
   * @return A new instance of the dynamic class.
   */  
  void *create();

  /** 
   * Deallocate a dynamic class.  WARNING the destructors are NOT called.
   * 
   * @param x A pointer the the dynamic class do be destroyed.
   */
  void destroy(void *x);

  /// @return The current total size of the dynamic class.
  unsigned int getClassSize() {return classSize;}

  /// @return The current number of nodes in this dynamic class factory.
  unsigned int getNumNodes() {return nodes.getSize();}

  /// @return The current number of existing instances of the dynamic class.
  unsigned int getNumClasses() {return numClasses;}

protected:
  /** 
   * This function is only callable by BasicDynamicClassNodeBase.
   * See BasicDynamicClassNodeBase::registerNode().
   */
  void registerNode(BasicDynamicClassNodeBase *node);

  friend class BasicDynamicClassNodeBase;
};
#endif
