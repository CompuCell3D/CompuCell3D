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

#ifndef CELLBOUNDARYTRACKERPLUGIN_H
#define CELLBOUNDARYTRACKERPLUGIN_H

#include <CompuCell3D/Plugin.h>

#include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
#include <BasicUtils/BasicClassAccessor.h>
#include <BasicUtils/BasicClassGroup.h> //had to include it to avoid problems with template instantiation
#include "CellBoundaryTracker.h"
#include <CompuCell3D/Field3D/AdjacentNeighbor.h>

namespace CompuCell3D {

  class Cell;
  class Field3DIndex;
  template <class T> class Field3D;


  class CellBoundaryTrackerPlugin : public Plugin, public CellGChangeWatcher {
    //CellBoundaryDynamicClassNode classNode; //will have to register it with cellFactory from Potts3D
//    SurfaceEnergy *surfaceEnergy;

    Field3D<CellG *> *cellFieldG;
    Dim3D fieldDim;
    //CellBoundaryTracker cellBoundaryTracker;
    BasicClassAccessor<CellBoundaryTracker> cellBoundaryTrackerAccessor;
    Simulator *simulator;
    bool periodicX,periodicY,periodicZ;
    
  public:
    CellBoundaryTrackerPlugin();
    virtual ~CellBoundaryTrackerPlugin();

    //CellBoundaryDynamicClassNode getClassNode() {return classNode;}

    // SimObject interface
    virtual void init(Simulator *simulator);

    // BCGChangeWatcher interface
    virtual void field3DChange(const Point3D &pt, CellG *newCell,
                               CellG *oldCell);

    // Begin XMLSerializable interface
    virtual void readXML(XMLPullParser &in);
    virtual void writeXML(XMLSerializer &out);
    virtual void initializeBoundaries()  ;
    BasicClassAccessor<CellBoundaryTracker> * getCellBoundaryTrackerAccessorPtr(){return & cellBoundaryTrackerAccessor;}
    // End XMLSerializable interface
    
   protected:
   double distance(double,double,double,double,double,double);
   
   virtual void testLatticeSanity();
   virtual void testLatticeSanityFull();
   bool isBoundaryPixel(Point3D pt);
   bool isTouchingLatticeBoundary(Point3D pt,Point3D ptAdj);
   bool watchingAllowed;
   AdjacentNeighbor adjNeighbor;
   long maxIndex; //maximum field index     
   long changeCounter;
  };
};
#endif
