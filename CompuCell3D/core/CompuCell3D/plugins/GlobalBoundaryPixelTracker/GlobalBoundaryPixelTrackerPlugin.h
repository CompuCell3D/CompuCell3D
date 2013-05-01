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

#ifndef GLOBALBOUNDARYPIXELTRACKERPLUGIN_H
#define GLOBALBOUNDARYPIXELTRACKERPLUGIN_H
#include <CompuCell3D/CC3D.h>
#include "GlobalBoundaryPixelTrackerDLLSpecifier.h"


class CC3DXMLElement;
namespace CompuCell3D {

  class Cell;
  class Field3DIndex;
  class Potts3D;
  template <class T> class Field3D;
  template <class T> class WatchableField3D;
  class BoundaryStrategy;

  
class GLOBALBOUNDARYPIXELTRACKER_EXPORT GlobalBoundaryPixelTrackerPlugin : public Plugin, public CellGChangeWatcher {

	ParallelUtilsOpenMP *pUtils;
	ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;

      //WatchableField3D<CellG *> *cellFieldG;
      Dim3D fieldDim;
      Simulator *simulator;
		Potts3D* potts;
		unsigned int maxNeighborIndex;
		BoundaryStrategy * boundaryStrategy;
		CC3DXMLElement *xmlData;
		std::set<Point3D> * boundaryPixelSetPtr;
			
   public:
      GlobalBoundaryPixelTrackerPlugin();
      virtual ~GlobalBoundaryPixelTrackerPlugin();
      
      
      // Field3DChangeWatcher interface
      virtual void field3DChange(const Point3D &pt, CellG *newCell,
                                 CellG *oldCell);
		
		//Plugin interface 
		virtual void init(Simulator *_simulator, CC3DXMLElement *_xmlData=0);
		virtual void extraInit(Simulator *_simulators);
		virtual void handleEvent(CC3DEvent & _event);		

		//Steerable interface
		virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
		virtual std::string steerableName();
		virtual std::string toString();

				
      
  };
};
#endif
