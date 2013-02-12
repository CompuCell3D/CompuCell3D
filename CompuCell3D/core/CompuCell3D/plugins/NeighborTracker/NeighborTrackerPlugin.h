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

#ifndef NEIGHBORTRACKERPLUGIN_H
#define NEIGHBORTRACKERPLUGIN_H
#include <CompuCell3D/CC3D.h>
// // // #include <CompuCell3D/Plugin.h>

// // // #include <CompuCell3D/Potts3D/Cell.h>
// // // #include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
// // // #include <BasicUtils/BasicClassAccessor.h>
// // // #include <BasicUtils/BasicClassGroup.h> //had to include it to avoid problems with template instantiation
// // // #include <PublicUtilities/ParallelUtilsOpenMP.h>
#include "NeighborTracker.h"

// // // #include <CompuCell3D/Field3D/AdjacentNeighbor.h>

#include "NeighborTrackerDLLSpecifier.h"


class CC3DXMLElement;
namespace CompuCell3D {

  class Cell;
  class Field3DIndex;
  template <class T> class Field3D;
  template <class T> class WatchableField3D;

  class CellInventory;
  class BoundaryStrategy;
  
class NEIGHBORTRACKER_EXPORT NeighborTrackerPlugin : public Plugin, public CellGChangeWatcher {

      ParallelUtilsOpenMP *pUtils;
      ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;
      WatchableField3D<CellG *> *cellFieldG;
      Dim3D fieldDim;
      BasicClassAccessor<NeighborTracker> neighborTrackerAccessor;
      Simulator *simulator;
      bool periodicX,periodicY,periodicZ;
      CellInventory * cellInventoryPtr;
      bool checkSanity;
      unsigned int checkFreq;

      unsigned int maxNeighborIndex;
      BoundaryStrategy *boundaryStrategy;

    
   public:
      NeighborTrackerPlugin();
      virtual ~NeighborTrackerPlugin();
      
      
      // Field3DChangeWatcher interface
      virtual void field3DChange(const Point3D &pt, CellG *newCell,
                                 CellG *oldCell);
      
		//Plugin interface 
		virtual void init(Simulator *_simulator, CC3DXMLElement *_xmlData=0);
		virtual std::string toString();



		BasicClassAccessor<NeighborTracker> * getNeighborTrackerAccessorPtr(){return & neighborTrackerAccessor;}
      // End XMLSerializable interface
      int returnNumber(){return 23432;}
	  short getCommonSurfaceArea(NeighborSurfaceData * _nsd){return _nsd->commonSurfaceArea;}
     

   protected:
      double distance(double,double,double,double,double,double);
      
      virtual void testLatticeSanityFull();
      bool isBoundaryPixel(Point3D pt);
      bool watchingAllowed;
      AdjacentNeighbor adjNeighbor;
      long maxIndex; //maximum field index
      long changeCounter;
  };
};
#endif
