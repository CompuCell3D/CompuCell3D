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

#ifndef VOLUMETRACKERPLUGIN_H
#define VOLUMETRACKERPLUGIN_H

#include <CompuCell3D/Plugin.h>
#include <CompuCell3D/Potts3D/Stepper.h>
#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
#include <CompuCell3D/Potts3D/Cell.h>
#include <PublicUtilities/ParallelUtilsOpenMP.h>
#include "VolumeTrackerDLLSpecifier.h"

#include <vector>


class CC3DXMLElement;

namespace CompuCell3D {
  class Potts3D;
  class CellG;
  class Simulator;

  


  class VOLUMETRACKER_EXPORT VolumeTrackerPlugin : public Plugin, public CellGChangeWatcher, public Stepper 
  {
	Potts3D *potts;
	CellG *deadCellG;
	Simulator *sim;
	ParallelUtilsOpenMP *pUtils;
	ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;
    
	std::vector<CellG *> deadCellVec; 



  public:
	VolumeTrackerPlugin();
	virtual ~VolumeTrackerPlugin();
	
	// SimObject interface
	virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);
	
	// CellChangeWatcher interface
	virtual void field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell);
	
	// Stepper interface
	virtual void step();
	virtual std::string toString();
	virtual std::string steerableName();
  };
};
#endif
