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

#include "BasicDynamicClassNodeBase.h"

#include "BasicDynamicClassFactory.h"
#include "BasicException.h"

void *BasicDynamicClassNodeBase::getNode(const void *x) const {
  ASSERT_OR_THROW("BasicDynamicClassNodeBase not registered",
		  factory != NULL);
  return &((unsigned char *)x)[offset];
}


void BasicDynamicClassNodeBase::
registerNode(BasicDynamicClassFactory *factory) {
  ASSERT_OR_THROW("BasicDynamicClassNodeBase already registered!",
		  this->factory == NULL);

  this->factory = factory;
  factory->registerNode(this);
}
