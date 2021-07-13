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

#ifndef FIELD3DCHANGEWATCHER_H
#define FIELD3DCHANGEWATCHER_H

#include "Point3D.h"


namespace CompuCell3D {

  template <class T>
  class Field3DChangeWatcher {
  public:
    /** 
     * Notify the watcher that a set is being performed.
     * This function will be called after the change has occured!
     * 
     * @param pt The location.
     * @param newValue The current value.
     * @param oldValue The previous value.
     */
    virtual void field3DChange(const Point3D &pt, T newValue,
			       T oldValue) = 0;
	
	/**
	* Notify the watcher that a set is being performed with respect to two points.
	* This function will be called after the change has occured!
	*
	* @param pt The location of the set.
	* @param addPt Another point with respect to the set.
	* @param newValue The current value.
	* @param oldValue The previous value.
	*/
	virtual void field3DChange(const Point3D &pt, const Point3D &addPt, T newValue, T oldValue) {};
  };
};
#endif
