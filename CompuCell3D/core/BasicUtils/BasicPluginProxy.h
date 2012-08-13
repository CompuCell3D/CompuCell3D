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

#ifndef BASICPLUGINPROXY_H
#define BASICPLUGINPROXY_H

#include <string>
#include <list>
#include <iostream>
#include <stdlib.h>

#include "BasicPluginManager.h"
#include "BasicPluginInfo.h"
#include "BasicException.h"
#include "BasicClassFactory.h"

template <class B, class T>
class BasicPluginProxy {
public:
  BasicPluginProxy(const std::string name, const std::string description, 
		   BasicPluginManager<B> *manager)
  {init(new BasicPluginInfo(name, description), manager);}

  BasicPluginProxy(const std::string name, const std::string description,
		   const unsigned int numDeps, const char *deps[],
		   BasicPluginManager<B> *manager)
  {init(new BasicPluginInfo(name, description, numDeps, deps), manager);}

  BasicPluginProxy(const BasicPluginInfo &info, BasicPluginManager<B> *manager)
  {init(new BasicPluginInfo(info), manager);}

protected:
  /** 
   * Initialize the class.  Called by constructors.
   * Throws a BasicException of manager is NULL.
   * 
   * @param manager A pointer to the Plugin Manager.
   */
  virtual void init(BasicPluginInfo *info, BasicPluginManager<B> *manager) {
    try {
      if (!manager) {
	std::cerr << "BasicPluginProxyBase() manager cannot be NULL!"
		  << std::endl;
	exit(1);
      }

      manager->registerPlugin(info, new BasicClassFactory<B, T>);

    } catch (BasicException &e) {
      manager->setPluginException(e);

    } catch (...) {
      manager->setPluginException
	(BasicException("Unknown exception during registration!"));
    }
  }
};

#endif
