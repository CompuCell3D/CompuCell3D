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
using namespace CompuCell3D;
using namespace std;

#include "UniformFieldInitializer.h"

UniformFieldInitializer::UniformFieldInitializer() :
	potts(0), sim(0) {}


void UniformFieldInitializer::init(Simulator *simulator, CC3DXMLElement * _xmlData) {
	sim = simulator;
	potts = simulator->getPotts();
	WatchableField3D<CellG *> *cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
	ASSERT_OR_THROW("initField() Cell field G cannot be null!", cellFieldG);
	Dim3D dim = cellFieldG->getDim();


	bool pluginAlreadyRegisteredFlag;
	Plugin *plugin = Simulator::pluginManager.get("VolumeTracker", &pluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded
	if (!pluginAlreadyRegisteredFlag)
		plugin->init(simulator);

	//clearing vector storing UniformFieldInitializerData (region definitions)
	initDataVec.clear();

	CC3DXMLElementList regionVec = _xmlData->getElements("Region");
	if (regionVec.size() > 0) {
		for (int i = 0; i < regionVec.size(); ++i) {
			UniformFieldInitializerData initData;

			if (regionVec[i]->findElement("Gap"))
				initData.gap = regionVec[i]->getFirstElement("Gap")->getUInt();
			if (regionVec[i]->findElement("Width"))
				initData.width = regionVec[i]->getFirstElement("Width")->getUInt();

			ASSERT_OR_THROW("UniformInitializer requires Types element inside Region section.See manual for details.", regionVec[i]->getFirstElement("Types"));
			initData.typeNamesString = regionVec[i]->getFirstElement("Types")->getText();
			parseStringIntoList(initData.typeNamesString, initData.typeNames, ",");

			if (regionVec[i]->findElement("BoxMax")) {
				initData.boxMax.x = regionVec[i]->getFirstElement("BoxMax")->getAttributeAsUInt("x");
				initData.boxMax.y = regionVec[i]->getFirstElement("BoxMax")->getAttributeAsUInt("y");
				initData.boxMax.z = regionVec[i]->getFirstElement("BoxMax")->getAttributeAsUInt("z");
			}

			if (regionVec[i]->findElement("BoxMin")) {
				initData.boxMin.x = regionVec[i]->getFirstElement("BoxMin")->getAttributeAsUInt("x");
				initData.boxMin.y = regionVec[i]->getFirstElement("BoxMin")->getAttributeAsUInt("y");
				initData.boxMin.z = regionVec[i]->getFirstElement("BoxMin")->getAttributeAsUInt("z");
			}


			initDataVec.push_back(initData);
		}
	}

}
void UniformFieldInitializer::layOutCells(const UniformFieldInitializerData & _initData) {

	int size = _initData.gap + _initData.width;
	int cellWidth = _initData.width;

	WatchableField3D<CellG *> *cellField = (WatchableField3D<CellG *> *)potts->getCellFieldG();
	ASSERT_OR_THROW("initField() Cell field cannot be null!", cellField);

	Dim3D dim = cellField->getDim();
	Point3D boxDim = _initData.boxMax - _initData.boxMin;
	cerr << " _initData.boxMin " << _initData.boxMin << " _initData.boxMax=" << _initData.boxMax << " dim=" << dim << endl;

	ASSERT_OR_THROW(" BOX DOES NOT FIT INTO LATTICE ",
		_initData.boxMin.x >= 0 && _initData.boxMin.y >= 0 && _initData.boxMin.z >= 0
		&& _initData.boxMax.x <= dim.x
		&& _initData.boxMax.y <= dim.y
		&& _initData.boxMax.z <= dim.z
		);

	Dim3D itDim;

	itDim.x = boxDim.x / size;
	if (boxDim.x % size) itDim.x += 1;
	itDim.y = boxDim.y / size;
	if (boxDim.y % size) itDim.y += 1;
	itDim.z = boxDim.z / size;
	if (boxDim.z % size) itDim.z += 1;

	cerr << "itDim=" << itDim << endl;
	Point3D pt;
	Point3D cellPt;
	CellG *cell;



	for (int z = 0; z < itDim.z; z++)
		for (int y = 0; y < itDim.y; y++)
			for (int x = 0; x < itDim.x; x++) {
				pt.x = _initData.boxMin.x + x * size;
				pt.y = _initData.boxMin.y + y * size;
				pt.z = _initData.boxMin.z + z * size;
				//cerr<<" pt="<<pt<<endl;

				if (BoundaryStrategy::getInstance()->isValid(pt)) {
					cell = potts->createCellG(pt);
					cell->type = initCellType(_initData);
					potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
					//It is necessary to do it this way because steppers are called only when we are performing pixel copies
					// but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
					//inventory unless you call steppers(VolumeTrackerPlugin) explicitely

				}
				else {
					continue;
				}

				for (cellPt.z = pt.z; cellPt.z < pt.z + cellWidth &&
					cellPt.z < dim.z; cellPt.z++)
					for (cellPt.y = pt.y; cellPt.y < pt.y + cellWidth &&
						cellPt.y < dim.y; cellPt.y++)
						for (cellPt.x = pt.x; cellPt.x < pt.x + cellWidth &&
							cellPt.x < dim.x; cellPt.x++) {

							if (BoundaryStrategy::getInstance()->isValid(pt))
								cellField->set(cellPt, cell);

						}
				potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
				//It is necessary to do it this way because steppers are called only when we are performing pixel copies
				// but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
				//inventory unless you call steppers(VolumeTrackerPlugin) explicitely
			}


}

unsigned char UniformFieldInitializer::initCellType(const UniformFieldInitializerData & _initData) {
	Automaton * automaton = potts->getAutomaton();
	if (_initData.typeNames.size() == 0) {//by default each newly created type will be 1 
		return 1;
	}
	else { //user has specified more than one cell type - will pick randomly the type
		BasicRandomNumberGenerator * randGen = BasicRandomNumberGenerator::getInstance();
		int index = randGen->getInteger(0, _initData.typeNames.size() - 1);


		return automaton->getTypeId(_initData.typeNames[index]);
	}

}

void UniformFieldInitializer::start() {
	if (sim->getRestartEnabled()) {
		return;  // we will not initialize cells if restart flag is on
	}
	cerr << "INSIDE START" << endl;

	WatchableField3D<CellG *> *cellField = (WatchableField3D<CellG *> *) potts->getCellFieldG();
	ASSERT_OR_THROW("initField() Cell field cannot be null!", cellField);
	Dim3D dim = cellField->getDim();




	if (initDataVec.size() != 0) {
		for (int i = 0; i < initDataVec.size(); ++i) {

			layOutCells(initDataVec[i]);
		}
	}

}

std::string UniformFieldInitializer::steerableName() {
	return toString();
}

std::string UniformFieldInitializer::toString() {
	return "UniformInitializer";
}





