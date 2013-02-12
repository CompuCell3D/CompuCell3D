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

#ifndef LENGTHCONSTRAINTLOCALFLEXPLUGIN_H
#define LENGTHCONSTRAINTLOCALFLEXPLUGIN_H

#include <CompuCell3D/CC3D.h>
// // // #include <CompuCell3D/Plugin.h>

// // // #include <CompuCell3D/Potts3D/EnergyFunction.h>
// // // #include <BasicUtils/BasicClassAccessor.h>
// // // #include <BasicUtils/BasicClassGroup.h> //had to include it to avoid problems with template instantiation

#include "LengthConstraintLocalFlexData.h"



// // // #include <CompuCell3D/Potts3D/CellGChangeWatcher.h>


// // // #include <CompuCell3D/Potts3D/Cell.h>
#include "LengthConstraintLocalFlexDLLSpecifier.h"


class CC3DXMLElement;
namespace CompuCell3D {
  
  class Potts3D;
 
  class CellG;
  class BoundaryStrategy;
  

  class LENGTHCONSTRAINTLOCALFLEX_EXPORT LengthConstraintLocalFlexPlugin : public Plugin,public EnergyFunction {
        
    

   Potts3D *potts;
	BasicClassAccessor<LengthConstraintLocalFlexData> lengthConstraintLocalFlexDataAccessor;
	BoundaryStrategy * boundaryStrategy;
    
  public:



    typedef double (LengthConstraintLocalFlexPlugin::*changeEnergyFcnPtr_t)(const Point3D &pt, const CellG *newCell,
                               const CellG *oldCell);

    LengthConstraintLocalFlexPlugin();
    virtual ~LengthConstraintLocalFlexPlugin();

    // SimObject interface
    virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);
	 virtual std::string toString();
    
    
    

	BasicClassAccessor<LengthConstraintLocalFlexData> * getLengthConstraintLocalFlexDataPtr(){return & lengthConstraintLocalFlexDataAccessor;}
	
	void setLengthConstraintData(CellG * _cell, double _lambdaLength, double _targetLength);    
	double getTargetLength(CellG * _cell);  
	double getLambdaLength(CellG * _cell);  
	

   //Energy Function interface
    virtual double changeEnergy(const Point3D &pt, const CellG *newCell,
				const CellG *oldCell);

    virtual double changeEnergy_xy(const Point3D &pt, const CellG *newCell,
				const CellG *oldCell);

    virtual double changeEnergy_xz(const Point3D &pt, const CellG *newCell,
				const CellG *oldCell);

    virtual double changeEnergy_yz(const Point3D &pt, const CellG *newCell,
				const CellG *oldCell);

   changeEnergyFcnPtr_t changeEnergyFcnPtr;

    


  };
};
#endif
