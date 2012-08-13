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

#ifndef ENERGYFUNCTION_H
#define ENERGYFUNCTION_H

#include "Potts3D.h"
#include <string>

namespace CompuCell3D {

	/** 
	* The Potts3D energy function interface.
	*/

	class Point3D;
	class CellG;

	class EnergyFunction {
	protected:
		//const Potts3D *potts3D;	

	public:
		EnergyFunction() {}
		virtual ~EnergyFunction() {}
	

		/**
		* Called by Potts3D when this function is registered.
		*/
		//virtual void registerPotts3D(Potts3D *potts3D) {
		//		this->potts3D = potts3D;
		//}

		/** 
		* @return The energy change for this function at point pt.
		*/
		virtual double localEnergy(const Point3D &pt){return 0.0;};

		/** 
		* @param pt The point of change.
		* @param newCell The new spin.
		* 
		* @return The energy change of changing point pt to newCell.
		*/
		//     virtual double changeEnergy(const Point3D &pt, const Cell *newCell,
		// 				const Cell *oldCell) = 0;

		virtual double changeEnergy(const Point3D &pt, const CellG *newCell,const CellG *oldCell) 
		{
			if(1!=1);return 0.0;
		}
		virtual std::string toString()
		{
			return std::string("EnergyFunction");
		}
	};
};
#endif
