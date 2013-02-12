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

#ifndef VISCOSITYPLUGIN_H
#define VISCOSITYPLUGIN_H

#include <CompuCell3D/CC3D.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
// // // #include <CompuCell3D/Potts3D/Cell.h>

// // // #include <BasicUtils/BasicString.h>
// // // #include <BasicUtils/BasicException.h>
// // // //#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
// // // #include <CompuCell3D/Potts3D/EnergyFunction.h>
// // // #include <CompuCell3D/Boundary/BoundaryStrategy.h>

#include "ViscosityDLLSpecifier.h"


// // // #include <CompuCell3D/Plugin.h>

class CC3DXMLElement ;
namespace CompuCell3D {

  class Potts3D;
  class CellG;
  class Simulator;
  class NeighborTracker;
  
  class VISCOSITY_EXPORT ViscosityPlugin : public Plugin , public EnergyFunction/*,public CellGChangeWatcher*/ {

  private:
    Potts3D *potts;
	CC3DXMLElement *xmlData;
    Simulator *sim;
    BasicClassAccessor<NeighborTracker> *neighborTrackerAccessorPtr;
   Point3D boundaryConditionIndicator;
   Dim3D fieldDim;
   BoundaryStrategy *boundaryStrategy;
   unsigned int maxNeighborIndex;
   double lambdaViscosity;
   std::string pluginName;
    
   double dist(double _x, double _y, double _z);
    
  public:
    ViscosityPlugin();
    virtual ~ViscosityPlugin();

    // SimObject interface
	virtual void extraInit(Simulator *simulator);
	virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);
    virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
	 
	//// CellChangeWatcher interface
	//virtual void field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell);
    
	 //EnergyFunction interface
	virtual double changeEnergy(const Point3D &pt, const CellG *newCell,const CellG *oldCell);
    virtual std::string steerableName();
	virtual std::string toString();
    
  };
};
#endif
