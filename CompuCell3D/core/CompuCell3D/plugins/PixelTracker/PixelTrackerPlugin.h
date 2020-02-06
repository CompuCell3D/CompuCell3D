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

#ifndef PIXELTRACKERPLUGIN_H
#define PIXELTRACKERPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "PixelTracker.h"
#include "PixelTrackerDLLSpecifier.h"


class CC3DXMLElement;
namespace CompuCell3D {

  class Cell;
  class Field3DIndex;
  template <class T> class Field3D;
  template <class T> class WatchableField3D;


  
class PIXELTRACKER_EXPORT PixelTrackerPlugin : public Plugin, public CellGChangeWatcher {

      //WatchableField3D<CellG *> *cellFieldG;
      Dim3D fieldDim;
      BasicClassAccessor<PixelTracker> pixelTrackerAccessor;
      Simulator *simulator;
	  Potts3D *potts;	

	  bool fullInitAtStart;
	  bool fullInitState;

	  std::set<PixelTrackerData> mediumPixelSet;
	  bool trackMedium;
	  virtual void fullMediumTrackerDataInit();
    
   public:
      PixelTrackerPlugin();
      virtual ~PixelTrackerPlugin();
      
      
      // Field3DChangeWatcher interface
      virtual void field3DChange(const Point3D &pt, CellG *newCell,
                                 CellG *oldCell);
      
		//Plugin interface 
		virtual void init(Simulator *_simulator, CC3DXMLElement *_xmlData=0);
		virtual std::string toString();
		virtual void handleEvent(CC3DEvent & _event);		

		BasicClassAccessor<PixelTracker> * getPixelTrackerAccessorPtr(){return & pixelTrackerAccessor;}
		//had to include this function to get set itereation working properly with Python , and Player that has restart capabilities
		PixelTrackerData * getPixelTrackerData(PixelTrackerData * _psd){return _psd;}

		virtual void enableFullInitAtStart(bool _fullInitAtStart = true) { fullInitAtStart = _fullInitAtStart; }
		virtual bool fullyInitialized() { return fullInitState; }

		virtual void fullTrackerDataInit(Point3D _ptChange = Point3D(-1, -1, -1), CellG *oldCell = 0);
		virtual void enableMediumTracker(bool _trackMedium = true) { trackMedium = _trackMedium; }
		virtual bool trackingMedium() { return trackMedium; }
		virtual std::set<PixelTrackerData> getMediumPixelSet() { return mediumPixelSet; }
      
  };
};
#endif
