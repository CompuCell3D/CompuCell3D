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

#ifndef CELLCM_H
#define CELLCM_H


#include "ComponentsDLLSpecifier.h"
#include <PublicUtilities/Vector3.h>

#ifndef PyObject_HEAD
struct _object; //forward declare
typedef _object PyObject; //type redefinition
#endif

class BasicClassGroup;

namespace CenterModel {

	/**
	* A Potts3D cell.
	*/

	class COMPONENTS_EXPORT CellCM{
	public:
		typedef unsigned char CellType_t;
		CellCM():
			id(0),
			type(0), 
			lookupIdx(-1),
			//x(0.0),y(0.0),z(0.0),
			radius(2.0),
			interactionRadius(1.0),
			mass(1.0),
			volume(0.0),
			surface(0.0)
		{}

		long id;
		CellType_t type;

		long lookupIdx;

		//double x,y,z;
        
        Vector3 position;
        
		double radius;
		double interactionRadius;
		double mass;

		double volume;

		double surface;

		void grow();

		BasicClassGroup *extraAttribPtr;

		PyObject *pyAttrib;
	};



};
#endif
