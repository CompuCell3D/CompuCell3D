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

#ifndef SIMPLEARRAYPLUGIN_H
#define SIMPLEARRAYPLUGIN_H

#include <CompuCell3D/Potts3D/Stepper.h>
#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
#include <CompuCell3D/Potts3D/Cell.h>

#include <BasicUtils/BasicClassAccessor.h>
#include <BasicUtils/BasicClassGroup.h> //had to include it to avoid problems with template instantiation

#include <CompuCell3D/Plugin.h>
#include "SimpleArray.h"


#include <vector>

#include <CompuCell3D/dllDeclarationSpecifier.h>

namespace CompuCell3D {
  class Potts3D;

  class Cell;
  class CellInventory;
  
  template <typename Y> class Field3DImpl;
  
  class DECLSPECIFIER SimpleArrayPlugin : public Plugin {
     
     BasicClassAccessor<SimpleArray> simpleArrayAccessor;
     CellInventory * cellInventoryPtr;
     Field3D<float> *simpleArrayFieldPtr;
     Potts3D *potts;
   

  public:
    SimpleArrayPlugin();
    virtual ~SimpleArrayPlugin();

    BasicClassAccessor<SimpleArray> * getSimpleArrayAccessorPtr(){return &simpleArrayAccessor;}
    // SimObject interface
    virtual void init(Simulator *_simulator, ParseData *_pd=0);
    virtual void extraInit(Simulator *_simulator);
    void update(ParseData *_pd);

    
    virtual void readXML(XMLPullParser &in);
    virtual void writeXML(XMLSerializer &out);
    virtual std::string toString(){return "SimpleArray";}
    
    // End XMLSerializable interface
     protected:
        std::vector<double> probMatrix;
   
  };
};
#endif
