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

#ifndef CONTACTPLUGIN_H
#define CONTACTPLUGIN_H

#include <CompuCell3D/CC3D.h>

#include "ContactDLLSpecifier.h"

class CC3DXMLElement;

namespace CompuCell3D {
	class Potts3D;
	class Automaton;
	class BoundaryStrategy;

	class CONTACT_EXPORT ContactPlugin : public Plugin,public EnergyFunction {

		Potts3D *potts;

		typedef std::unordered_map<unsigned char, std::unordered_map<unsigned char, double> > contactEnergyArray_t;

		contactEnergyArray_t contactEnergyArray;

		std::string autoName;
		double depth;

		Automaton *automaton;
		bool weightDistance;
		unsigned int maxNeighborIndex;
		BoundaryStrategy * boundaryStrategy;
		CC3DXMLElement *xmlData;

	public:
		ContactPlugin();
		virtual ~ContactPlugin();
		
		//Plugin interface

		virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);
		virtual void extraInit(Simulator *simulator);

		//EnergyFunction Interface
		
		virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

		double contactEnergy(const CellG *cell1, const CellG *cell2);

		/**
		* Sets the contact energy for two cell types.  A -1 type is interpreted
		* as the medium.
		*/
		void setContactEnergy(const std::string typeName1,
			const std::string typeName2, const double energy);


		//Steerable interface

		virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
		virtual std::string steerableName();
		virtual std::string toString();

	};

	inline double ContactPlugin::contactEnergy(const CellG *cell1, const CellG *cell2) {
		return contactEnergyArray[cell1 ? cell1->type : 0][cell2 ? cell2->type : 0];
	}
};
#endif
