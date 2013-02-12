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


#include <CompuCell3D/CC3D.h>
// // // #include <CompuCell3D/Potts3D/CellInventory.h>
// // // #include <CompuCell3D/Automaton/Automaton.h>

// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Potts3D/Cell.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
// // // #include <CompuCell3D/Field3D/Point3D.h>
// // // #include <CompuCell3D/Field3D/Dim3D.h>
using namespace CompuCell3D;


// // // #include <BasicUtils/BasicString.h>
// // // #include <BasicUtils/BasicException.h>

// // // #include <BasicUtils/BasicRandomNumberGenerator.h>
// // // #include <string>

#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>
// // // #include <BasicUtils/BasicClassAccessor.h>
// // // #include <fstream>
// // // #include <sstream>

using namespace std;



#include "FoamDataOutput.h"

FoamDataOutput::FoamDataOutput() :
potts(0),
neighborTrackerAccessorPtr(0),
surFlag(false),
volFlag(false),
numNeighborsFlag(false),
cellIDFlag(false)
{}


void FoamDataOutput::init(Simulator *_simulator, CC3DXMLElement *_xmlData) 
{
	potts = _simulator->getPotts();
	cellInventoryPtr = & potts->getCellInventory();
	CC3DXMLElement *outputXMLElement=_xmlData->getFirstElement("Output");

	ASSERT_OR_THROW("You need to provide Output element to FoamDataOutput Steppable with at least file name", outputXMLElement);

	if(outputXMLElement)
	{
		if(outputXMLElement->findAttribute("FileName"))
			fileName=outputXMLElement->getAttribute("FileName");
	
		if(outputXMLElement->findAttribute("Volume"))
			volFlag=true;
		
		if(outputXMLElement->findAttribute("Surface"))
			surFlag=true;

		if(outputXMLElement->findAttribute("NumberOfNeighbors"))
			numNeighborsFlag=true;

		if(outputXMLElement->findAttribute("CellID"))
			numNeighborsFlag=cellIDFlag;
	}
}

void FoamDataOutput::extraInit(Simulator *simulator) 
{
	if(numNeighborsFlag)
	{
		bool pluginAlreadyRegisteredFlag;
		NeighborTrackerPlugin * neighborTrackerPluginPtr=(NeighborTrackerPlugin*)(Simulator::pluginManager.get("NeighborTracker",&pluginAlreadyRegisteredFlag));
		if (!pluginAlreadyRegisteredFlag)      
			neighborTrackerPluginPtr->init(simulator);
		ASSERT_OR_THROW("NeighborTracker plugin not initialized!", neighborTrackerPluginPtr);
		neighborTrackerAccessorPtr=neighborTrackerPluginPtr->getNeighborTrackerAccessorPtr();
		ASSERT_OR_THROW("neighborAccessorPtr  not initialized!", neighborTrackerAccessorPtr);
	}
}


void FoamDataOutput::start() {}

void FoamDataOutput::step(const unsigned int currentStep) 
{
	ostringstream str;
	str<<fileName<<"."<<currentStep;
	ofstream out(str.str().c_str());

	//    cerr<<"cellIDFlag= " << cellIDFlag<<" numNeighborsFlag="<<numNeighborsFlag<<" volFlag="<<volFlag<<" surFlag="<<surFlag<<endl;

	CellInventory::cellInventoryIterator cInvItr;
	CellG * cell;
	std::set<NeighborSurfaceData > * neighborData;

	for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr )
	{
		cell=cellInventoryPtr->getCell(cInvItr);
		//cell=*cInvItr;
		if(cellIDFlag)
			out<<cell->id<<"\t";

		if(volFlag)
			out<<cell->volume<<"\t";

		if(surFlag)
			out<<cell->surface<<"\t";

		if(numNeighborsFlag){
			neighborData = &(neighborTrackerAccessorPtr->get(cell->extraAttribPtr)->cellNeighbors);
			out<<neighborData->size()<<"\t";
		}
		out<<endl;
	}
}


std::string FoamDataOutput::toString(){
	return "FoamDataOutput";
}





