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

#ifndef FOAMDATAOUTPUT_H
#define FOAMDATAOUTPUT_H

#include <CompuCell3D/CC3D.h>

// // // #include <CompuCell3D/Steppable.h>
// // // #include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/plugins/NeighborTracker/NeighborTracker.h>
#include "FoamDataOutputDLLSpecifier.h"
#include <string>

template <typename Y> class BasicClassAccessor;

namespace CompuCell3D {
  class Potts3D;
  class CellInventory;
   
  class FOAMDATAOUTPUT_EXPORT FoamDataOutput : public Steppable 
  {
	Potts3D *potts;
	CellInventory * cellInventoryPtr;
	Dim3D dim;
	BasicClassAccessor<NeighborTracker> * neighborTrackerAccessorPtr;
	std::string fileName;
	bool surFlag;
	bool volFlag;
	bool numNeighborsFlag;
	bool cellIDFlag;

  public:
	FoamDataOutput();
	virtual ~FoamDataOutput(){};  
	void setPotts(Potts3D *potts) {this->potts = potts;}
	
	// SimObject interface
	virtual void init(Simulator *_simulator, CC3DXMLElement *_xmlData=0);
	virtual void extraInit(Simulator *simulator);
	virtual std::string toString();
	// Begin Steppable interface
	virtual void start();
	virtual void step(const unsigned int currentStep);
	virtual void finish() {}
	// End Steppable interface
  };
};
#endif
