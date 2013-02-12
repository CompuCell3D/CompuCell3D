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

#ifndef MOMENTOFINERTIAPLUGIN_H
#define MOMENTOFINERTIAPLUGIN_H

#include <CompuCell3D/CC3D.h>
// // // #include <CompuCell3D/Plugin.h>




// // // #include <CompuCell3D/Field3D/Point3D.h>
// // // #include <CompuCell3D/Field3D/Dim3D.h>

// // // #include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
// // // #include <CompuCell3D/Potts3D/Cell.h>
// // // #include <BasicUtils/BasicClassAccessor.h>
// // // #include <BasicUtils/BasicClassGroup.h> //had to include it to avoid problems with template instantiation
// // // #include <cmath>
// // // #include <vector>
#define roundf(a) ((fmod(a,1)<0.5)?floor(a):ceil(a))


#include "MomentOfInertiaDLLSpecifier.h"

namespace CompuCell3D {
  class Potts3D;
  class ParseData;
  class MomentOfInertiaPlugin;
  class BoundaryStrategy; 
  
  

  class MOMENTOFINERTIA_EXPORT MomentOfInertiaPlugin : public Plugin, public CellGChangeWatcher {
    Potts3D *potts;
	 Simulator *simulator;
   Point3D boundaryConditionIndicator;
   Dim3D fieldDim;
	BoundaryStrategy *boundaryStrategy;
	int lastMCSPrintedWarning;
   
  public:
	    typedef void (MomentOfInertiaPlugin::*field3DChangeFcnPtr_t)(const Point3D &pt,  CellG *newCell,CellG *oldCell);

	    //typedef void (MomentOfInertiaPlugin::*getSemiaxesFctPtr_t)(CellG *_cell,double & _majorAxis , double & _medianAxis, double &_minorAxis);
		typedef std::vector<double>(MomentOfInertiaPlugin::*getSemiaxesFctPtr_t)(CellG *_cell);

    MomentOfInertiaPlugin();
    virtual ~MomentOfInertiaPlugin();
    
    void getMomentOfInertia(CellG *cell, float I[3]) const
    {
      ASSERT_OR_THROW("getMomentOfInertia() Cell cannot be NULL!", cell);

      unsigned int volume = cell->volume;
      ASSERT_OR_THROW("getMomentOfInertia() Cell volume is 0!", volume);

      

      I[0] = cell->iXX;
      I[1] = cell->iYY;
      I[2] = cell->iZZ;

    }

  
    // SimObject interface
    virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);

	

    // BCGChangeWatcher interface
    virtual void field3DChange(const Point3D &pt, CellG *newCell,
                                CellG *oldCell);

    virtual void cellOrientation_xy(const Point3D &pt, CellG *newCell,
                                CellG *oldCell);

    virtual void cellOrientation_xz(const Point3D &pt, CellG *newCell,
                                CellG *oldCell);

    virtual void cellOrientation_yz(const Point3D &pt, CellG *newCell,
                                CellG *oldCell);

	field3DChangeFcnPtr_t cellOrientationFcnPtr;

	getSemiaxesFctPtr_t getSemiaxesFctPtr;

	//convenience functions for accessing semiaxes
	//void getSemiaxes(CellG *_cell,double & _majorAxis , double & _medianAxis, double &_minorAxis);
	//void getSemiaxesXY(CellG *_cell,double & _majorAxis , double & _medianAxis, double &_minorAxis);
	//void getSemiaxesXZ(CellG *_cell,double & _majorAxis , double & _medianAxis, double &_minorAxis);
	//void getSemiaxesYZ(CellG *_cell,double & _majorAxis , double & _medianAxis, double &_minorAxis);
	//void getSemiaxes3D(CellG *_cell,double & _majorAxis , double & _medianAxis, double &_minorAxis);

	std::vector<double> getSemiaxes(CellG *_cell);
	std::vector<double> getSemiaxesXY(CellG *_cell);
	std::vector<double> getSemiaxesXZ(CellG *_cell);
	std::vector<double> getSemiaxesYZ(CellG *_cell);
	std::vector<double> getSemiaxes3D(CellG *_cell);


   virtual std::string toString();

  };
};
#endif
