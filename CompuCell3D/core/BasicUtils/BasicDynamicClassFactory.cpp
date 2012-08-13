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

#include "BasicDynamicClassFactory.h"

#include "BasicDynamicClassNodeBase.h"

#include <stdlib.h>

BasicDynamicClassFactory::BasicDynamicClassFactory() :
  classSize(0), numClasses(0) {
}

void *BasicDynamicClassFactory::create() {
  numClasses++;

  // malloc 0 is not safe
  void *x = malloc(classSize ? classSize : 1);

  for (unsigned int i = 0; i < nodes.getSize(); i++)
    nodes[i]->_init(x);

  return x;
}

void BasicDynamicClassFactory::destroy(void *x) {
  numClasses--;
  free(x);
}

void BasicDynamicClassFactory::registerNode(BasicDynamicClassNodeBase *node) {
  node->setOffset(classSize);
  classSize += node->getSize();
  nodes.put(node);
}
