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

#ifndef CENTEROFMASSMONITOR_H
#define CENTEROFMASSMONITOR_H

 
#include <CompuCell3D/Field3D/Dim3D.h>
#include <CA/CACellFieldChangeWatcher.h>



#define roundf(a) ((fmod(a,1)<0.5)?floor(a):ceil(a))


#include "CenterOfMassMonitorDLLSpecifier.h"

namespace CompuCell3D {
  class CACell;
  class CAManager;
  class BoundaryStrategy;  

  template<typename T>
  class Field3D;  


  class CENTEROFMASSMONITOR_EXPORT CenterOfMassMonitor : public CACellFieldChangeWatcher {
    
    
  private:
   Point3D boundaryConditionIndicator;
   Dim3D fieldDim;
   BoundaryStrategy *boundaryStrategy;
   CAManager *caManager;
   Field3D<CACell *> * cellField;





  public:
    CenterOfMassMonitor();
    virtual ~CenterOfMassMonitor();
    
	void init(CAManager *_caManager);

    // BCGChangeWatcher interface
    virtual void field3DChange(const Point3D &pt, CACell *newCell,
                                CACell *oldCell);

    // virtual std::string toString();
	// virtual std::string steerableName();
  };
};
#endif
