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

#ifndef BASICCLASSGROUPFACTORY_H
#define BASICCLASSGROUPFACTORY_H

#include "BasicArray.h"
#include "BasicClassFactoryBase.h"
#include "BasicClassGroup.h"

class BasicClassAccessorBase;

/** 
 * Manages (de)allocation of groups of classes in a type safe manner.
 *
 * Pros:
 *  <ul>
 *   <li>Classes can be dynamically added to the group after compile time.</li>
 *   <li>Unlike BasicDynamicClassFactory constructors and destructors are 
 *       called.</li>
 *   <li>Unlike BasicDynamicClassFactory virtual classes are ok.</li>
 *  </ul>
 *
 * Cons: 
 *  <ul>
 *   <li>Used more memory than BasicDynamicClassFactory because a list of
 *       pointers to the classes is kept in each BasicClassGroup.</li>
 *   <li>If you want to use a constructor with arguments you must override both
 *       BasicClassFactory::create() and
 *       BasicClassAccessor::createClassFactory() 
 *       with your own implementations.<li>
 *  </ul>
 *
 * See also BasicDynamicClassFactory.
 */
class BasicClassGroupFactory : public BasicClassFactoryBase<BasicClassGroup> {
  BasicArray<BasicClassFactoryBase<void> *> classFactories;
  BasicArray<BasicClassAccessorBase *> classAccessors;

public:
  virtual ~BasicClassGroupFactory();

  /** 
   * Register a new class with the group.
   * BasicClassAccessor::createClassFactory() is
   * called to get an instance of the class factory. BasicClassGroupFactory 
   * will deallocate this factory when it is destructed.
   * 
   * @param accessor The accessor for the class.
   */
  void registerClass(BasicClassAccessorBase *accessor);

  /** 
   * The constructors of each of the classes in the group will be called in 
   * the order they where registered with the factory.
   *
   * @return A new instance of the class group.
   */
  virtual BasicClassGroup *create();

  /** 
   * The destructors of each of the classes in the group will be called in the 
   * order they were registered with the factory.
   *
   * @param group The class group to be deallocated.
   */
  virtual void destroy(BasicClassGroup *group);
};

#endif
