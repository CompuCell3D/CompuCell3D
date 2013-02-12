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

#ifndef NEIGHBORSTICKPLUGIN_H
#define NEIGHBORSTICKPLUGIN_H
#include <CompuCell3D/CC3D.h>

// // // #include <CompuCell3D/Potts3D/EnergyFunction.h>

// // // #include <CompuCell3D/Plugin.h>

// // // #include <BasicUtils/BasicClassAccessor.h>
// // // #include <BasicUtils/BasicClassGroup.h> //had to include it to avoid problems with template instantiation
#include <CompuCell3D/plugins/NeighborTracker/NeighborTracker.h>

// // // #include <map>
// // // #include <vector>

#include "NeighborStickDLLSpecifier.h"

class CC3DXMLElement;

namespace CompuCell3D {
  

    class Potts3D;
  class Automaton;
  class Simulator;

  class NEIGHBORSTICK_EXPORT NeighborStickPlugin : public Plugin, public EnergyFunction  {

    //energyFunction data
    Potts3D *potts;
    Simulator *sim;
    BasicClassAccessor<NeighborTracker> * neighborTrackerAccessorPtr;
    
    double thresh;
    std::vector<std::string> typeNames;
    std::string typeNamesString;
    std::vector<int> idNames;
    
    void Types(std::string  _type){
       typeNames.push_back(_type);
    }
    
    Automaton *automaton;
  
  
	 CC3DXMLElement *xmlData;

  public:
    NeighborStickPlugin();
    virtual ~NeighborStickPlugin();


	 //plugin interface
    virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);
	 virtual void extraInit(Simulator *simulator);

	 //EnergyFunction interface
    virtual double changeEnergy(const Point3D &pt, const CellG *newCell,
                                const CellG *oldCell);


    //Steerable interface
    virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
    virtual std::string steerableName();
	 virtual std::string toString();

  };
};
#endif
