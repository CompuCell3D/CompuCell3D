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

#ifndef SIMPLECLOCKPLUGIN_H
#define SIMPLECLOCKPLUGIN_H

#include <CompuCell3D/CC3D.h>
// // // #include <CompuCell3D/Plugin.h>


// // // #include <CompuCell3D/Potts3D/Stepper.h>

// // // #include <CompuCell3D/Potts3D/CellGChangeWatcher.h>

// // // #include <CompuCell3D/Potts3D/Cell.h>
// // // #include <BasicUtils/BasicClassAccessor.h>
// // // #include <BasicUtils/BasicClassGroup.h> //had to include it to avoid problems with template instantiation
#include "SimpleClock.h"


#include "SimpleClockDLLSpecifier.h"

namespace CompuCell3D {
  class Potts3D;

  class Cell;
  template <typename Y> class Field3DImpl;
  
  class SIMPLECLOCK_EXPORT SimpleClockPlugin : public Plugin
  //, public CellGChangeWatcher,public Stepper 
		       {
    
    BasicClassAccessor<SimpleClock> simpleClockAccessor;
        
    
    Field3D<float> *simpleClockFieldPtr;
    Potts3D *potts;
        
	// Point3D pt;
  public:

    SimpleClockPlugin();
    virtual ~SimpleClockPlugin();
	 
    BasicClassAccessor<SimpleClock> * getSimpleClockAccessorPtr(){return &simpleClockAccessor;}
    // SimObject interface
    virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);

    void setSimpleClockFieldPtr( Field3D<float> *_simpleClockFieldPtr){simpleClockFieldPtr=_simpleClockFieldPtr;}   

  };
};
#endif
