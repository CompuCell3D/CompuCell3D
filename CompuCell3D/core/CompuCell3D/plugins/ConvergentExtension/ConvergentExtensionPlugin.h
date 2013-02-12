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

#ifndef CONVERGENTEXTENSIONPLUGIN_H
#define CONVERGENTEXTENSIONPLUGIN_H

#include <CompuCell3D/CC3D.h>

// // // #include <CompuCell3D/Potts3D/EnergyFunction.h>
// // // #include <CompuCell3D/Plugin.h>
// // // #include <map>
// // // #include <set>
// // // #include <vector>

#include "ConvergentExtensionDLLSpecifier.h"

class CC3DXMLElement;

namespace CompuCell3D {
	class Potts3D;
	class Automaton;
	class BoundaryStrategy;

	class CONVERGENTEXTENSION_EXPORT ConvergentExtensionPlugin : public Plugin,public EnergyFunction {

		Potts3D *potts;

		std::set<unsigned char> interactingTypes ;
	   std::vector<double> alphaConvExtVec;
		std::map<std::string , double> typeNameAlphaConvExtMap;
		unsigned char maxTypeId;

		double depth;

		Automaton *automaton;
		
		unsigned int maxNeighborIndex;
		BoundaryStrategy * boundaryStrategy;
		CC3DXMLElement *xmlData;

	public:
		ConvergentExtensionPlugin();
		virtual ~ConvergentExtensionPlugin();
		//Plugin interface
		virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);
		virtual void extraInit(Simulator *simulator);
		
		

		//EnergyFunction Interface
		virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

		//Steerable interface
		virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
		virtual std::string steerableName();
		virtual std::string toString();


	};
};
#endif
