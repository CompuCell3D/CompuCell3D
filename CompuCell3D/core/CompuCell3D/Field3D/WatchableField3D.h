/*************************************************************************
 *    CompuCell - A software framework for multimodel simulations of     *
 * biocomplexity problems Copyright (C) 2003 University of Notre Dame,   *
 *                             Indiana                                   *
 *                                                                       *
 * This program is free software; IF YOU AGREE TO CITE USE OF CompuCell  *
 *  IN ALL RELATED RESEARCH PUBLICATIONS according to the terms of the   *
 *  CompuCell GNU General Public License RIDER you can redistribute it   *
 * and/or modify it under the terms of the GNU General Public License as *
 *  published by the Free Software Foundation; either version 2 of the   *
 *         License, or (at your option) any later version.               *
 *                                                                       *
 * This program is distributed in the hope that it will be useful, but   *
 *      WITHOUT ANY WARRANTY; without even the implied warranty of       *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    *
 *             General Public License for more details.                  *
 *                                                                       *
 *  You should have received a copy of the GNU General Public License    *
 *     along with this program; if not, write to the Free Software       *
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.        *
 *************************************************************************/

#ifndef WATCHABLEFIELD3D_H
#define WATCHABLEFIELD3D_H

#include "Field3DImpl.h"
#include "Field3DChangeWatcher.h"

#include <BasicUtils/BasicArray.h>
#include <BasicUtils/BasicException.h>

namespace CompuCell3D {

  template <class T>
  class Field3DImpl;

  template <class T>
  class WatchableField3D: public Field3DImpl<T> {
    BasicArray<Field3DChangeWatcher<T> *> changeWatchers;

  public:
    /** 
     * @param dim The field dimensions
     * @param initialValue The initial value of all data elements in the field.
     */
    WatchableField3D(const Dim3D dim, const T &initialValue) : 
      Field3DImpl<T>(dim, initialValue) {}

      virtual ~WatchableField3D(){}   
    virtual void addChangeWatcher(Field3DChangeWatcher<T> *watcher) {
      ASSERT_OR_THROW("addChangeWatcher() watcher cannot be NULL!", watcher);
      changeWatchers.put(watcher);
    }

    virtual void set(const Point3D &pt, const T value) {
      T oldValue = Field3DImpl<T>::get(pt);
      Field3DImpl<T>::set(pt, value);

      for (unsigned int i = 0; i < changeWatchers.getSize(); i++)
	changeWatchers[i]->field3DChange(pt, value, oldValue);
    }

	virtual void set(const Point3D &pt, const Point3D &addPt, const T value) {
		T oldValue = Field3DImpl<T>::get(pt);
		Field3DImpl<T>::set(pt, value);

		for (unsigned int i = 0; i < changeWatchers.getSize(); i++) {
			changeWatchers[i]->field3DChange(pt, value, oldValue);
			changeWatchers[i]->field3DChange(pt, addPt, value, oldValue);
		}
	}
  };
};
#endif
