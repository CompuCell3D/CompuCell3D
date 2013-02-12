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

//Author: Margriet Palm CWI, Netherlands

#include <CompuCell3D/CC3D.h>

// // // #include <CompuCell3D/Automaton/Automaton.h>
// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Potts3D/Cell.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
// // // #include <CompuCell3D/Field3D/Point3D.h>
// // // #include <CompuCell3D/Field3D/Dim3D.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>

using namespace CompuCell3D;

// // // #include <BasicUtils/BasicRandomNumberGenerator.h>
// // // #include <PublicUtilities/StringUtils.h>

// // // #include <string>
// // // #include <map>
// // // #include <cmath>
// // // #include <vector>
// // // #include <algorithm>

#include "RandomBlobInitializer.h"

using namespace std;

RandomBlobInitializer::RandomBlobInitializer():
    mit(0),
    potts(0),
    simulator(0),
    rand(0),
    cellField(0),
    pixelTrackerAccessorPtr(0),
    builder(0),    
    cellInventoryPtr(0)
{
	ndiv,growsteps = 0;
	borderTypeID = -1;
	showStats=false;
}

RandomBlobInitializer::~RandomBlobInitializer(){
    delete builder;
}

void RandomBlobInitializer::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {
	cout << "START randomblob\n";
	simulator = _simulator;
	potts = _simulator->getPotts();
	cellField = (WatchableField3D<CellG *> *)potts->getCellFieldG();
	ASSERT_OR_THROW("initField() Cell field G cannot be null!", cellField);
	dim=cellField->getDim();
	cellInventoryPtr = & potts->getCellInventory();
	builder = new FieldBuilder(_simulator);
    
    update(_xmlData,true);
	
    
}

void RandomBlobInitializer::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
    setParameters(simulator,_xmlData);
}

void RandomBlobInitializer::setParameters(Simulator *_simulator, CC3DXMLElement *_xmlData){
	// initiate random generator
	rand = BasicRandomNumberGenerator::getInstance();
	if(_xmlData->getFirstElement("seed"))
		rand->setSeed(_xmlData->getFirstElement("seed")->getInt());
	builder->setRandomGenerator(rand);
	// set builder boxes
	Dim3D boxMin = Dim3D(0,0,0);
	Dim3D boxMax = cellField->getDim();
	if(_xmlData->getFirstElement("offset")){
		int offsetX = _xmlData->getFirstElement("offset")->getAttributeAsUInt("x");
		int offsetY = _xmlData->getFirstElement("offset")->getAttributeAsUInt("y");
		int offsetZ = _xmlData->getFirstElement("offset")->getAttributeAsUInt("z");
		boxMin=Dim3D(offsetX,offsetY,offsetZ);
		boxMax=Dim3D(dim.x-offsetX,dim.y-offsetY,dim.z-offsetZ);
	}
	builder->setBoxes(boxMin,boxMax);
	int order = 1;
//	cout << "read order\n";
	if(_xmlData->getFirstElement("order"))
		order = _xmlData->getFirstElement("order")->getInt();
//	cout << "order = " << order << endl;
	if (order == 2){builder->setNeighborListSO();}
	else {builder->setNeighborListFO();}
	// read types and set bias
	vector<string> typeNames;
	vector<string> biasVec;
	// read number of divisions
	if(_xmlData->getFirstElement("ndiv"))
		ndiv =_xmlData->getFirstElement("ndiv")->getInt();
	if(_xmlData->getFirstElement("types")){
		string typeNamesString = _xmlData->getFirstElement("types")->getText();
		parseStringIntoList(typeNamesString,typeNames, ",");
	}
	bool biasSet = false;
	if(_xmlData->getFirstElement("bias")){
		string biasString = _xmlData->getFirstElement("bias")->getText();
		parseStringIntoList(biasString,biasVec, ",");
		if (biasVec.size() == typeNames.size()){
			builder->setTypeVec(pow((double)2,(int)ndiv),typeNames,biasVec);
			biasSet = true;
		}
	}
	if (!biasSet)
		builder->setTypeVec(pow((double)2,(int)ndiv),typeNames);
	// read number of growsteps
	if(_xmlData->getFirstElement("growsteps"))
		growsteps = _xmlData->getFirstElement("growsteps")->getInt();

	// get initial blobsize (before eden growth)
	blobsize = Dim3D(0,0,0);
	if(_xmlData->getFirstElement("initBlobSize")){
		blobsize.x = _xmlData->getFirstElement("initBlobSize")->getAttributeAsUInt("x");
		blobsize.y = _xmlData->getFirstElement("initBlobSize")->getAttributeAsUInt("y");
		blobsize.z = _xmlData->getFirstElement("initBlobSize")->getAttributeAsUInt("z");
	}
	blobpos = Dim3D(dim.x/2,dim.y/2,dim.z/2);
	if(_xmlData->getFirstElement("blobPos")){
		blobpos.x = _xmlData->getFirstElement("blobPos")->getAttributeAsUInt("x");
		blobpos.y = _xmlData->getFirstElement("blobPos")->getAttributeAsUInt("y");
		blobpos.z = _xmlData->getFirstElement("blobPos")->getAttributeAsUInt("z");
	}
	// get   type
	Automaton * automaton=potts->getAutomaton();
	if(_xmlData->getFirstElement("borderType")){
		borderTypeID = automaton->getTypeId(_xmlData->getFirstElement("borderType")->getText());
	}
	// check showstats
	if(_xmlData->getFirstElement("showStats"))
		showStats = true;
}

void RandomBlobInitializer::extraInit(Simulator *simulator){
//	cout << "EXTRA INIT BLOBINITIALIZER\n";
	bool pluginAlreadyRegisteredFlag;
	mit = (MitosisSteppable*)(Simulator::steppableManager.get("Mitosis",&pluginAlreadyRegisteredFlag));
	if (!pluginAlreadyRegisteredFlag){
		mit->init(simulator);
	}
	ASSERT_OR_THROW("MitosisSteppable not initialized!", mit);
}

void RandomBlobInitializer::start(){
	Dim3D pos;
	if ((blobsize.x*blobsize.y*blobsize.z)==1)
		pos = blobpos;
	else{
		pos.x = (blobpos.x > blobsize.x/2) ? blobpos.x-blobsize.x/2 : 0;
		pos.y = (blobpos.y > blobsize.y/2) ? blobpos.y-blobsize.y/2 : 0;
		pos.z = (blobpos.z > blobsize.z/2) ? blobpos.z-blobsize.z/2 : 0;
	}
	builder->addCell(pos,blobsize);
	//~ cout << "grow cell\n";
	builder->growCells(growsteps);
	//~ cout << "divide cell\n";
	for (int i=0; i<ndiv; i++)
		divide();
	if (borderTypeID >= 0)
		builder->addBorderCell(borderTypeID);
	if (showStats){ builder->showCellStats(borderTypeID);}
}

void RandomBlobInitializer::divide(){
	CellInventory::cellInventoryIterator cInvItr;
	CellG * cell;
	CellG * child;
	PixelTracker * pixelTracker;
	set<PixelTrackerData>::iterator pixelItr;
	Point3D pt;
	vector<CellG*> cells;
	for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr ){
		if (cellInventoryPtr->getCell(cInvItr)->volume > 2)
			cells.push_back(cellInventoryPtr->getCell(cInvItr));
	}
	if ((int)cells.size() > 0){
		vector<CellG*>::iterator it;
		for (it=cells.begin(); it < cells.end(); it++){
			mit->doDirectionalMitosisAlongMinorAxis(*it);
			if (mit->childCell)
				builder->setType(mit->childCell);
		}
	}
	else{ cout << "cells are too small, not dividing\n";}
}

std::string RandomBlobInitializer::toString(){
   return "RandomBlobInitializer";
}


std::string RandomBlobInitializer::steerableName(){
   return toString();
}




