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

#ifndef CONNECTIVITYLOCALFLEXPLUGIN_H
#define CONNECTIVITYLOCALFLEXPLUGIN_H

 #include <CompuCell3D/CC3D.h>
#include "ConnectivityLocalFlexData.h"

// // // #include <CompuCell3D/Potts3D/EnergyFunction.h>

// // // #include <CompuCell3D/Plugin.h>

// // // #include <BasicUtils/BasicClassAccessor.h>
// // // #include <BasicUtils/BasicClassGroup.h> //had to include it to avoid problems with template instantiation
#include "ConnectivityLocalFlexDLLSpecifier.h"


namespace CompuCell3D {
  
  class CellG;
  class Potts3D;
  class Automaton;
  class BoundaryStrategy;
  class Simulator;

  class CONNECTIVITYLOCALFLEX_EXPORT  ConnectivityLocalFlexPlugin : public Plugin,public EnergyFunction {
  
	BasicClassAccessor<ConnectivityLocalFlexData> connectivityLocalFlexDataAccessor;
	 //Energy Function data
    Potts3D *potts;

    // std::vector<CellG*> uniqueCells;    
    // double penalty;

    // bool mediumFlag;
    unsigned int numberOfNeighbors;
    std::vector<int> offsetsIndex; //this vector will contain indexes of offsets in the neighborListVector so that accessing
    //them using offsetsindex will ensure correct clockwise ordering
    //std::vector<Point3D> n;
    unsigned int maxNeighborIndex;
    BoundaryStrategy * boundaryStrategy;

  public:
    ConnectivityLocalFlexPlugin();
    virtual ~ConnectivityLocalFlexPlugin();

	 //EnergyFunction interface
    virtual double changeEnergy(const Point3D &pt, const CellG *newCell,
                                const CellG *oldCell);
	//Plugin interface
    virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);

	BasicClassAccessor<ConnectivityLocalFlexData> * getConnectivityLocalFlexDataPtr(){return & connectivityLocalFlexDataAccessor;}
	void setConnectivityStrength(CellG * _cell, double _connectivityStrength);
	double getConnectivityStrength(CellG * _cell);
	
	//Steerable interface
	virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
	virtual std::string steerableName();
	virtual std::string toString();

	//EnergyFunction methods
  protected:
    /**
     * @return The index used for ordering connectivity energies in the map.
     */
    void addUnique(CellG*,std::vector<CellG*> &);
    void initializeNeighborsOffsets();
    void orderNeighborsClockwise(Point3D & _midPoint, const std::vector<Point3D> & _offsets, std::vector<int> & _offsetsIndex);




  };
};
#endif
