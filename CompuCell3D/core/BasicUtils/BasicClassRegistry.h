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

#ifndef BASICCLASSREGISTRY_H
#define BASICCLASSREGISTRY_H

#include "BasicException.h"
#include "BasicClassFactoryBase.h"
#include "BasicSmartPointer.h"

#include <string>
#include <map>

template <class T>
class BasicClassRegistry {
  typedef std::map<std::string, BasicClassFactoryBase<T> *> factoryMap_t;
  factoryMap_t factoryMap;

public:
  BasicClassRegistry() {}

  ~BasicClassRegistry() {
    typename factoryMap_t::iterator it;
    for (it = factoryMap.begin(); it != factoryMap.end(); it++)
      delete it->second;
  }

  BasicClassFactoryBase<T> *unregisterFactory(const std::string id) {
    BasicClassFactoryBase<T> *factory = factoryMap[id];
    factoryMap.erase(id);
    return factory;
  }

  void registerFactory(BasicClassFactoryBase<T> *factory,
		       const std::string id) {
    ASSERT_OR_THROW(std::string("registerFactory() Factory with id '") +
		    id + "' already registered",
		    factoryMap.find(id) == factoryMap.end());
    
    factoryMap[id] = factory;
  }

  BasicSmartPointer<T> create(const std::string id) {
    BasicClassFactoryBase<T> *factory = factoryMap[id];
    ASSERT_OR_THROW(std::string("create() Factory '") + id + "' not found!",
		    factory);
    
    return BasicSmartPointer<T>(factory->create());
  }
};

#endif
