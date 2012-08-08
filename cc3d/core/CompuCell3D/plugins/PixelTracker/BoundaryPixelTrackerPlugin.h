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

#ifndef BOUNDARYPIXELTRACKERPLUGIN_H
#define BOUNDARYPIXELTRACKERPLUGIN_H

#include <CompuCell3D/Plugin.h>

#include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
#include <BasicUtils/BasicClassAccessor.h>
#include <BasicUtils/BasicClassGroup.h> //had to include it to avoid problems with template instantiation
#include "BoundaryPixelTracker.h"
#include <CompuCell3D/Field3D/AdjacentNeighbor.h>

#include "PixelTrackerDLLSpecifier.h"


class CC3DXMLElement;
namespace CompuCell3D {

  class Cell;
  class Field3DIndex;
  class Potts3D;
  template <class T> class Field3D;
  template <class T> class WatchableField3D;
  class BoundaryStrategy;

  
class PIXELTRACKER_EXPORT BoundaryPixelTrackerPlugin : public Plugin, public CellGChangeWatcher {

      //WatchableField3D<CellG *> *cellFieldG;
      Dim3D fieldDim;
      BasicClassAccessor<BoundaryPixelTracker> boundaryPixelTrackerAccessor;
      Simulator *simulator;
		Potts3D* potts;
		unsigned int maxNeighborIndex;
		BoundaryStrategy * boundaryStrategy;
		CC3DXMLElement *xmlData;

		
   public:
      BoundaryPixelTrackerPlugin();
      virtual ~BoundaryPixelTrackerPlugin();
      
      
      // Field3DChangeWatcher interface
      virtual void field3DChange(const Point3D &pt, CellG *newCell,
                                 CellG *oldCell);

      
		//Plugin interface 
		virtual void init(Simulator *_simulator, CC3DXMLElement *_xmlData=0);
		virtual void extraInit(Simulator *_simulators);


		//Steerable interface
		virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
		virtual std::string steerableName();
		virtual std::string toString();


		BasicClassAccessor<BoundaryPixelTracker> * getBoundaryPixelTrackerAccessorPtr(){return & boundaryPixelTrackerAccessor;}
		//had to include this function to get set inereation working properly with Python , and Player that has restart capabilities
		BoundaryPixelTrackerData * getBoundaryPixelTrackerData(BoundaryPixelTrackerData * _psd){return _psd;}

      
  };
};
#endif
