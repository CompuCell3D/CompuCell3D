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

#ifndef CONNECTIVITYGLOBALPLUGIN_H
#define CONNECTIVITYGLOBALPLUGIN_H

 #include <CompuCell3D/CC3D.h>

#include "ConnectivityGlobalData.h"


// // // #include <CompuCell3D/Potts3D/EnergyFunction.h>
// // // #include <CompuCell3D/Plugin.h>

// // // #include <BasicUtils/BasicClassAccessor.h>
// // // #include <BasicUtils/BasicClassGroup.h> //had to include it to avoid problems with template instantiation


#include "ConnectivityGlobalDLLSpecifier.h"
// // // #include <vector>

class CC3DXMLElement;

namespace CompuCell3D {
  class Potts3D;
  class Automaton;
  class BoundaryStrategy;


  class CONNECTIVITYGLOBAL_EXPORT ConnectivityGlobalPlugin : public Plugin,public EnergyFunction {

    //Energy Function data
  private:

    BasicClassAccessor<ConnectivityGlobalData> connectivityGlobalDataAccessor;
  
    unsigned int maxNeighborIndex;
	unsigned int max_neighbor_index_local_search;
    BoundaryStrategy * boundaryStrategy;
    
	 Potts3D *potts;
	 std::vector<double> penaltyVec;
	 unsigned char maxTypeId;
	 bool doNotPrecheckConnectivity;
	 bool fast_algorithm;
    
	 typedef double (ConnectivityGlobalPlugin::*changeEnergyFcnPtr_t)(const Point3D &pt, const CellG *newCell,const CellG *oldCell);
	 changeEnergyFcnPtr_t changeEnergyFcnPtr;



  public:
    ConnectivityGlobalPlugin();
    virtual ~ConnectivityGlobalPlugin();

	 //Plugin interface
    virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);

    BasicClassAccessor<ConnectivityGlobalData> * getConnectivityGlobalDataPtr(){return & connectivityGlobalDataAccessor;}
	void setConnectivityStrength(CellG * _cell, double _connectivityStrength);
	double getConnectivityStrength(CellG * _cell);


	 virtual std::string toString();

	 //EnergyFunction interface
	     virtual double changeEnergy(const Point3D &pt, const CellG *newCell,
                                const CellG *oldCell);

		 double changeEnergyLegacy(const Point3D &pt, const CellG *newCell,
			 const CellG *oldCell);

		 double changeEnergyFast(const Point3D &pt, const CellG *newCell,
			 const CellG *oldCell);

		 
	bool checkIfCellIsFragmented(const CellG * cell,Point3D cellPixel);
	bool check_local_connectivity(const Point3D &pt, const CellG *cell, unsigned int max_neighbor_index_local_search, bool add_pt_to_bfs);

    //SteerableObject interface
    virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
    virtual std::string steerableName();


  };
};
#endif
