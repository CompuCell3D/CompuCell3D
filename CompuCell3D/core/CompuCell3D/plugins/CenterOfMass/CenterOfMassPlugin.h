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

#ifndef CENTEROFMASSPLUGIN_H
#define CENTEROFMASSPLUGIN_H

 #include <CompuCell3D/CC3D.h>

#define roundf(a) ((fmod(a,1)<0.5)?floor(a):ceil(a))


#include "CenterOfMassDLLSpecifier.h"

namespace CompuCell3D {
  class Potts3D;
  class ParseData;
  class BoundaryStrategy;

  

  class CENTEROFMASS_EXPORT CenterOfMassPlugin : public Plugin, public CellGChangeWatcher {
    Potts3D *potts;
    
  private:
   Point3D boundaryConditionIndicator;
   Dim3D fieldDim;
   BoundaryStrategy *boundaryStrategy;
   //determine allowed ranges of COM position - it COM is outside those values we will shift COM (by multiples of lattice sizes +1,-1*lattice size) to inside the allowed area
   Coordinates3D<double> allowedAreaMin;
   Coordinates3D<double> allowedAreaMax;



  public:
    CenterOfMassPlugin();
    virtual ~CenterOfMassPlugin();
    
    void getCenterOfMass(CellG *cell, float cm[3]) const
    {
      ASSERT_OR_THROW("getCenterOfMass() Cell cannot be NULL!", cell);

      unsigned int volume = cell->volume;
      ASSERT_OR_THROW("getCenterOfMass() Cell volume is 0!", volume);

      

      cm[0] = cell->xCM / (float)volume;
      cm[1] = cell->yCM / (float)volume;
      cm[2] = cell->zCM / (float)volume;

    }

    void getCenterOfMass(CellG *cell, float & _x, float & _y, float & _z) const
    {
      ASSERT_OR_THROW("getCenterOfMass() Cell cannot be NULL!", cell);

      unsigned int volume = cell->volume;
      ASSERT_OR_THROW("getCenterOfMass() Cell volume is 0!", volume);

      

      _x = cell->xCM / (float)volume;
      _y = cell->yCM / (float)volume;
      _z = cell->zCM / (float)volume;

    }

        
    Point3D getCenterOfMass(CellG *cell) const{
      
      float floatCM[3];
      getCenterOfMass(cell, floatCM);
      
      return Point3D((unsigned short)roundf(floatCM[0]),
            (unsigned short)roundf(floatCM[1]),
            (unsigned short)roundf(floatCM[2]));

    }
    // BCGChangeWatcher interface
     void field3DCheck(const Point3D &pt, CellG *newCell,
                                CellG *oldCell);
	 virtual void handleEvent(CC3DEvent & _event);

	//void updateCOMsAfterLatticeShift(Dim3D _shiftVec);
    // SimObject interface
    //virtual void init(Simulator *simulator, ParseData *_pd=0);
	 virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);

    // BCGChangeWatcher interface
    virtual void field3DChange(const Point3D &pt, CellG *newCell,
                                CellG *oldCell);

   virtual std::string toString();
	virtual std::string steerableName();
  };
};
#endif
