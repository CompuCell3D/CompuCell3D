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

// // // #include <CompuCell3D/Automaton/Automaton.h>
// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Potts3D/Cell.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
// // // #include <CompuCell3D/Field3D/Point3D.h>
// // // #include <CompuCell3D/Field3D/Dim3D.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>
#include <CompuCell3D/plugins/PixelTracker/PixelTracker.h>
#include <CompuCell3D/plugins/PixelTracker/PixelTrackerPlugin.h>
using namespace CompuCell3D;

// // // #include <BasicUtils/BasicRandomNumberGenerator.h>
// // // #include <PublicUtilities/StringUtils.h>

// // // #include <string>
// // // #include <map>

#include "RandomFieldInitializer.h"

using namespace std;

RandomFieldInitializer::RandomFieldInitializer():    
    potts(0),
    simulator(0),
    rand(0),
    cellField(0),
    builder(0)
{

	ncells,growsteps = 0;
	borderTypeID = -1;
	showStats=false;
}

RandomFieldInitializer::~RandomFieldInitializer(){
    delete builder;
}

void RandomFieldInitializer::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {
	simulator = _simulator;
	potts = _simulator->getPotts();
	cellField = (WatchableField3D<CellG *> *)potts->getCellFieldG();
	ASSERT_OR_THROW("initField() Cell field G cannot be null!", cellField);
	dim=cellField->getDim();
	builder = new FieldBuilder(_simulator);
	// setParameters(_simulator,_xmlData);
    update(_xmlData,true);
}



void RandomFieldInitializer::extraInit(Simulator *simulator){}

void RandomFieldInitializer::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
    setParameters(simulator,_xmlData);
}

void RandomFieldInitializer::setParameters(Simulator *_simulator, CC3DXMLElement *_xmlData){
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
		boxMax.x = dim.x-offsetX;
		boxMax.y = dim.y-offsetY;
		boxMax.z = dim.z-offsetZ;
	}
	builder->setBoxes(boxMin,boxMax);
	int order = 1;
	if(_xmlData->getFirstElement("order"))
		order = _xmlData->getFirstElement("order")->getInt();
	cout << "order = " << order << endl;
	if (order == 2){builder->setNeighborListSO();}
	else {builder->setNeighborListFO();}
	// read types and set bias
	vector<string> typeNames;
	vector<string> biasVec;
	if(_xmlData->getFirstElement("types")){
		string typeNamesString = _xmlData->getFirstElement("types")->getText();
		parseStringIntoList(typeNamesString,typeNames, ",");
	}
	// read number of growsteps
	if(_xmlData->getFirstElement("growsteps"))
		growsteps = _xmlData->getFirstElement("growsteps")->getInt();
	// read number of cells
	if(_xmlData->getFirstElement("ncells"))
		ncells=_xmlData->getFirstElement("ncells")->getInt();
	bool biasSet = false;
	if(_xmlData->getFirstElement("bias")){
		string biasString = _xmlData->getFirstElement("bias")->getText();
		parseStringIntoList(biasString,biasVec, ",");
		if (biasVec.size() == typeNames.size()){
			builder->setTypeVec(ncells,typeNames,biasVec);
			biasSet = true;
		}
	}
	if (!biasSet)
		builder->setTypeVec(ncells,typeNames);
	if (ncells > (boxMax.x*boxMax.y*boxMax.z)){
		ncells = boxMax.x*boxMax.y*boxMax.z;
		growsteps = 1;
		cout << "#########################\n";
		cout << "Too much cells!\nncells is set to " << ncells << endl;
		cout << "growsteps is set to 0\n";
		cout << "#########################\n";
	}
	// get border type
	Automaton * automaton=potts->getAutomaton();
	if(_xmlData->getFirstElement("borderType")){
		borderTypeID = automaton->getTypeId(_xmlData->getFirstElement("borderType")->getText());
	}
	// check showstats
	if(_xmlData->getFirstElement("showStats"))
		showStats = true;
}

void RandomFieldInitializer::start(){
	int i;
	for (i=0;i<ncells;i++){
		builder->addCell();
	}
	builder->growCells(growsteps);
	if (borderTypeID >= 0){
		builder->addBorderCell(borderTypeID);
	}
	if (showStats){ builder->showCellStats(borderTypeID);}
}

std::string RandomFieldInitializer::toString(){
   return "RandomFieldInitializer";
}


std::string RandomFieldInitializer::steerableName(){
   return toString();
}
