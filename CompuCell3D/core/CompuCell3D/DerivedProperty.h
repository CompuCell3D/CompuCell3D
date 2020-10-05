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
#ifndef DERIVEDPROPERTY_H
#define DERIVEDPROPERTY_H

namespace CompuCell3D {

	/**
	DerivedProperty: Derived properties that mimic the behavior of Python properties in C++
	Written by T.J. Sego, Ph.D.
	9/12/2020

	The DerivedProperty returns the value of the property on demand according to a function 
	that defines the DerivedProperty. All requisite information to calculate the current 
	value of a DerivedProperty must be intrinsic to the instance of the parent class.

	For CC3D Python support, follow the procedure of existing implementations 
	using the SWIG macro DERIVEDPROPERTYEXTENSORPY in CompuCell3D/core/pyinterface/CompuCellPython/DerivedProperty.i
	*/

	template <typename ParentType, typename PropertyType, PropertyType(ParentType::*PropertyFunction)()>
	class DerivedProperty {

		// Parent object with this derived property as a member
		ParentType *obj;

	public:

		DerivedProperty() {}
		DerivedProperty(ParentType *_obj) :
			obj(_obj)
		{}
		~DerivedProperty() { obj = 0; }

		// Pretend to be a value
		operator PropertyType() const { return (obj->*PropertyFunction)(); }

	};

}

#endif