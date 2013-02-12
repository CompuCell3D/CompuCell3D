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
//

#include <CompuCell3D/CC3D.h>

// // // #include <CompuCell3D/Potts3D/CellInventory.h>
// // // #include <CompuCell3D/Automaton/Automaton.h>

// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Potts3D/Cell.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
// // // #include <CompuCell3D/Field3D/Point3D.h>
// // // #include <CompuCell3D/Field3D/Dim3D.h>
// // // #include <CompuCell3D/Field3D/Field3D.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>
using namespace CompuCell3D;


// // // #include <BasicUtils/BasicString.h>
// // // #include <BasicUtils/BasicException.h>
// // // #include <BasicUtils/BasicRandomNumberGenerator.h>

// // // #include <string>
using namespace std;


#include "DictyFieldInitializer.h"

DictyFieldInitializer::DictyFieldInitializer() :
potts(0), gotAmoebaeFieldBorder(false),presporeRatio(0.5), gap(1),width(2),amoebaeFieldBorder(10)
{}


void DictyFieldInitializer::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

	if(_xmlData->findElement("Gap"))
		gap=_xmlData->getFirstElement("Gap")->getUInt();

	if(_xmlData->findElement("Width"))
		width=_xmlData->getFirstElement("Width")->getUInt();

	if(_xmlData->findElement("AmoebaeFieldBorder"))
		amoebaeFieldBorder=_xmlData->getFirstElement("AmoebaeFieldBorder")->getUInt();

	if(_xmlData->findElement("ZonePoint")){
		zonePoint.x=_xmlData->getFirstElement("ZonePoint")->getAttributeAsUInt("x");
		zonePoint.y=_xmlData->getFirstElement("ZonePoint")->getAttributeAsUInt("y");
		zonePoint.z=_xmlData->getFirstElement("ZonePoint")->getAttributeAsUInt("z");
		zoneWidth=_xmlData->getFirstElement("ZonePoint")->getUInt();
	}

	if(_xmlData->findElement("PresporeRatio")){
		presporeRatio=_xmlData->getFirstElement("PresporeRatio")->getDouble();
		ASSERT_OR_THROW("Ratio must belong to [0,1]!",0<=presporeRatio && presporeRatio<=1.0);
	}



}

void DictyFieldInitializer::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

	update(_xmlData,true);

	potts = simulator->getPotts();
	automaton=potts->getAutomaton();
	cellField = (WatchableField3D<CellG*> *)potts->getCellFieldG();
	ASSERT_OR_THROW("initField() Cell field cannot be null!", cellField);
	ASSERT_OR_THROW("Could not find Center of Mass plugin",Simulator::pluginManager.get("CenterOfMass"));

	dim = cellField->getDim();

	if(!gotAmoebaeFieldBorder){
		amoebaeFieldBorder=dim.x;
	}


}

void DictyFieldInitializer::start() {

	// TODO: Chage this code so it write the 0 spins too.  This will make it
	//       possible to re-initialize a previously used field.

	int size = gap + width;



	//  CenterOfMassPlugin * comPlugin=(CenterOfMassPlugin*)(Simulator::pluginManager.get("CenterOfMass"));
	//  Cell *c;

	//  comPlugin->getCenterOfMass(c);

	Dim3D itDim;

	itDim.x = dim.x / size;
	if (dim.x % size) itDim.x += 1;
	itDim.y = dim.y / size;
	if (dim.y % size) itDim.y += 1;
	itDim.z = dim.z / size;
	if (dim.z % size) itDim.z += 1;

	Point3D pt;
	Point3D cellPt;
	CellG *cell;

	///this is only temporary initializer. Later I will rewrite it so that the walls and ground are thiner
	///preparing ground layer
	pt.x = 0;
	pt.y = 0;
	pt.z = 0;

	cell = potts->createCellG(pt);
	groundCell=cell;
	for (cellPt.z = pt.z; cellPt.z <= pt.z + 1*(width+gap)-1 && cellPt.z < dim.z; cellPt.z++)
		for (cellPt.y = pt.y; cellPt.y < pt.y + dim.y && cellPt.y < dim.y; cellPt.y++)
			for (cellPt.x = pt.x; cellPt.x < pt.x + dim.y && cellPt.x < dim.x; cellPt.x++)
				cellField->set(cellPt, cell);


	///walls
	cell = potts->createCellG(pt);
	wallCell=cell;
	for (cellPt.z = pt.z;   cellPt.z < dim.z; cellPt.z++)
		for (cellPt.y = pt.y;  cellPt.y < dim.y; cellPt.y++)
			for (cellPt.x = pt.x;  cellPt.x < dim.x; cellPt.x++){

				if(
					(int)fabs(1.0*cellPt.z-dim.z) % dim.z <=1.0 ||
					(int)fabs(1.0*cellPt.y-dim.y) % dim.y <=1.0 ||
					(int)fabs(1.0*cellPt.x-dim.x) % dim.x<=1.0
					){
						//cerr<<"wall at pt="<<cellPt<<endl;
						cellField->set(cellPt, cell);
				}

				/*         if(cellPt.z<width)///additionally thick wall at the bottom to prevent the diffusion into too large area
				cellField->set(cellPt, cell);*/
			}



			///laying down layer of aboebae - keeping the distance from the walls
			for (int z = 1; z < 2; z++)
				for (int y = 1; y < itDim.y-1; y++)
					for (int x = 1; x < itDim.x-1; x++) {

						pt.x = x * size;
						pt.y = y * size;
						pt.z = z * size;



						if(pt.x<amoebaeFieldBorder && pt.y<amoebaeFieldBorder ){
							cell = potts->createCellG(pt);


							for (cellPt.z = pt.z; cellPt.z < pt.z + width && cellPt.z < dim.z; cellPt.z++)
								for (cellPt.y = pt.y; cellPt.y < pt.y + width && cellPt.y < dim.y; cellPt.y++)
									for (cellPt.x = pt.x; cellPt.x < pt.x + width && cellPt.x < dim.x; cellPt.x++)
										cellField->set(cellPt, cell);
						}
					} 


					//Now will initialize types of cells
					initializeCellTypes();

					///preparing water layer
					/*   pt.x = 0;
					pt.y = 0;
					pt.z = width+gap;

					cell = 0;

					bool initializedWater=false; 
					//groundCell=cell;
					for (cellPt.z = pt.z; cellPt.z <= pt.z + width && cellPt.z < dim.z; cellPt.z++)
					for (cellPt.y = pt.y; cellPt.y < pt.y + dim.y && cellPt.y < dim.y; cellPt.y++)
					for (cellPt.x = pt.x; cellPt.x < pt.x + dim.y && cellPt.x < dim.x; cellPt.x++){


					if(!cellField->get(cellPt)){///check if medium
					if( ! initializedWater){ ///put water
					//cerr<<"CREATING WATER CELL "<<cellPt<<endl;
					cell=potts->createCellG(pt);
					cell->type=automaton->getTypeId("Water");
					cellField->set(cellPt,cell);
					initializedWater=true;

					}else{///put water
					//cerr<<"SETTING WATER CELL "<<cellPt<<endl;
					cellField->set(cellPt,cell);
					}
					}

					}*/



}


void DictyFieldInitializer::initializeCellTypes(){

	BasicRandomNumberGenerator *rand = BasicRandomNumberGenerator::getInstance();
	cellInventoryPtr=& potts->getCellInventory();

	///will initialize cell type here depending on the position of the cells
	CellInventory::cellInventoryIterator cInvItr;
	///loop over all the cells in the inventory
	Point3D com;
	CellG * cell;


	float x,y,z;

	for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr ){

		cell=cellInventoryPtr->getCell(cInvItr);
		//cell=*cInvItr;
		if(cell==groundCell)
			cell->type=automaton->getTypeId("Ground");
		else if(cell==wallCell){
			cell->type=automaton->getTypeId("Wall");
		}   
		else{

			com.x=cell->xCM/ cell->volume ;
			com.y=cell->yCM/ cell->volume;
			com.z=cell->zCM/ cell->volume;

			cerr<<"belongToZone(com)="<<belongToZone(com)<<" com="<<com<<endl;
			if(belongToZone(com)){
				cell->type=automaton->getTypeId("Autocycling");
				cerr<<"setting autocycling type="<<(int)cell->type<<endl;
			}else{
				if(rand->getRatio()<presporeRatio){
					cell->type=automaton->getTypeId("Prespore");
				}else{
					cell->type=automaton->getTypeId("Prestalk");
				}

			}


		}


	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool DictyFieldInitializer::belongToZone(Point3D com){

	if(
		com.x>zonePoint.x && com.x< (zonePoint.x+zoneWidth) &&
		com.y>zonePoint.y && com.y< (zonePoint.y+zoneWidth) &&
		com.z>zonePoint.z && com.z< (zonePoint.z+zoneWidth)
		)
		return true;
	else
		return false;

}



std::string DictyFieldInitializer::toString(){
	return "DictyInitializer";
}

std::string DictyFieldInitializer::steerableName(){
	return toString();
}

