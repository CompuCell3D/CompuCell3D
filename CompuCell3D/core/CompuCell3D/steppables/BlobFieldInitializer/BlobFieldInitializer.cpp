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
#include <CompuCell3D/plugins/CellType/CellTypePlugin.h>
// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Potts3D/Cell.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
// // // #include <CompuCell3D/Field3D/Point3D.h>
// // // #include <CompuCell3D/Field3D/Dim3D.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>
using namespace CompuCell3D;

//#include <XMLCereal/XMLPullParser.h>
//#include <XMLCereal/XMLSerializer.h>

// // // #include <BasicUtils/BasicString.h>
// // // #include <BasicUtils/BasicException.h>

// // // #include <BasicUtils/BasicClassGroup.h>
// // // #include <BasicUtils/BasicRandomNumberGenerator.h>
// // // #include <CompuCell3D/Potts3D/CellInventory.h>
// // // #include <CompuCell3D/plugins/CellType/CellTypePlugin.h>
// // // #include <PublicUtilities/StringUtils.h>
// // // #include <XMLUtils/CC3DXMLElement.h>

// // // #include <string>

// // // #include <math.h>

// // // #include <iostream>
using namespace std;


#include "BlobFieldInitializer.h"


std::string BlobFieldInitializer::steerableName(){
	return toString();
}

std::string BlobFieldInitializer::toString(){
	return "BlobInitializer";
}


BlobFieldInitializer::BlobFieldInitializer() :
potts(0),sim(0){}

void BlobFieldInitializer::init(Simulator *simulator,  CC3DXMLElement * _xmlData){

	sim=simulator;
	potts = simulator->getPotts();   
	WatchableField3D<CellG *> *cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
	ASSERT_OR_THROW("initField() Cell field G cannot be null!", cellFieldG);
	Dim3D dim = cellFieldG->getDim();


	bool pluginAlreadyRegisteredFlag;
	Plugin *plugin=Simulator::pluginManager.get("VolumeTracker",&pluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded
	if(!pluginAlreadyRegisteredFlag)
		plugin->init(simulator);

	if(_xmlData->getFirstElement("Radius")){
		oldStyleInitData.radius=_xmlData->getFirstElement("Radius")->getUInt();
		cerr<<"Got FE This Radius: "<<oldStyleInitData.radius<<endl;
		ASSERT_OR_THROW("Radius has to be greater than 0 and 2*radius cannot be bigger than lattice dimension x", oldStyleInitData.radius>0 && 2*(oldStyleInitData.radius)<(dim.x-2));
	}

	if(_xmlData->getFirstElement("Width")){
		oldStyleInitData.width=_xmlData->getFirstElement("Width")->getUInt();
		cerr<<"Got FE This Width: "<<oldStyleInitData.width<<endl;
	}
	if(_xmlData->getFirstElement("Gap")){
		oldStyleInitData.gap=_xmlData->getFirstElement("Gap")->getUInt();
		cerr<<"Got FE This Gap: "<<oldStyleInitData.gap<<endl;
	}



	if (_xmlData->getFirstElement("CellSortInit")){
		if(_xmlData->getFirstElement("CellSortInit")->getText()=="yes" ||_xmlData->getFirstElement("CellSortInit")->getText()=="Yes"){
			cellSortInit=true;
			cerr<<"SET CELLSORT INIT"<<endl;
		}
	}



	CC3DXMLElement *elem=_xmlData->getFirstElement("Engulfment");
	if (elem){
		engulfmentData.engulfment=true;
		engulfmentData.bottomType=elem->getAttribute("BottomType");
		engulfmentData.topType=elem->getAttribute("TopType");
		engulfmentData.engulfmentCutoff=elem->getAttributeAsUInt("EngulfmentCutoff");
		engulfmentData.engulfmentCoordinate=elem->getAttribute("EngulfmentCoordinate");
	}


	//clearing vector storing BlobFieldInitializerData (region definitions)
	blobInitializerData.clear();

	CC3DXMLElementList regionVec=_xmlData->getElements("Region");
	

	for (int i = 0 ; i<regionVec.size(); ++i){
		BlobFieldInitializerData initData;
		ASSERT_OR_THROW("BlobInitializer requires Radius element inside Region section.See manual for details.",regionVec[i]->getFirstElement("Radius"));
		initData.radius=regionVec[i]->getFirstElement("Radius")->getUInt();
		if (regionVec[i]->getFirstElement("Gap")){
			initData.gap=regionVec[i]->getFirstElement("Gap")->getUInt();
		}

		if (regionVec[i]->getFirstElement("Width")){
			initData.width=regionVec[i]->getFirstElement("Width")->getUInt();
		}

		ASSERT_OR_THROW("BlobInitializer requires Types element inside Region section.See manual for details.",regionVec[i]->getFirstElement("Types"));
		initData.typeNamesString=regionVec[i]->getFirstElement("Types")->cdata;

		parseStringIntoList(initData.typeNamesString , initData.typeNames , ",");

		ASSERT_OR_THROW("BlobInitializer requires Center element inside Region section.See manual for details.",regionVec[i]->getFirstElement("Center"));

		initData.center.x=regionVec[i]->getFirstElement("Center")->getAttributeAsUInt("x");
		initData.center.y=regionVec[i]->getFirstElement("Center")->getAttributeAsUInt("y");
		initData.center.z=regionVec[i]->getFirstElement("Center")->getAttributeAsUInt("z");

		cerr<<"radius="<<initData.radius<<" gap="<<initData.gap<<" types="<<initData.typeNamesString<<endl;
		blobInitializerData.push_back(initData);
	}






	cerr<<"GOT HERE BEFORE EXIT"<<endl;

}



double BlobFieldInitializer::distance(double ax, double ay, double az, double bx, double by, double bz) {
	//   printf("%f\n",sqrt((double)(ax - bx) * (ax - bx) + 
	//               (double)(ay - by) * (ay - by) + 
	//                      (double)(az - bz) * (az - bz)));
	return sqrt((double)(ax - bx) * (ax - bx) + 
		(double)(ay - by) * (ay - by) + 
		(double)(az - bz) * (az - bz));
}

void BlobFieldInitializer::layOutCells(const BlobFieldInitializerData & _initData){

	int size = _initData.gap + _initData.width;
	int cellWidth=_initData.width;

	WatchableField3D<CellG *> *cellField =(WatchableField3D<CellG *> *) potts->getCellFieldG();
	ASSERT_OR_THROW("initField() Cell field cannot be null!", cellField);

	Dim3D dim = cellField->getDim();

	ASSERT_OR_THROW("Radius has to be greater than 0 and 2*radius cannot be bigger than lattice dimension x", _initData.radius>0 && 2*_initData.radius<(dim.x-2));



	//  CenterOfMassPlugin * comPlugin=(CenterOfMassPlugin*)(Simulator::pluginManager.get("CenterOfMass"));
	//  Cell *c;

	//  comPlugin->getCenterOfMass(c);

	//   Dim3D itDim;
	// 
	//   itDim.x = boxDim.x / size;
	//   if (boxDim.x % size) itDim.x += 1;
	//   itDim.y = boxDim.y / size;
	//   if (boxDim.y % size) itDim.y += 1;
	//   itDim.z = boxDim.z / size;
	//   if (boxDim.z % size) itDim.z += 1;



	Dim3D itDim=getBlobDimensions(dim,size);
	cerr<<"itDim="<<itDim<<endl;


	Point3D pt;
	Point3D cellPt;
	CellG *cell;

	for (int z = 0; z < itDim.z; z++)
		for (int y = 0; y < itDim.y; y++)
			for (int x = 0; x < itDim.x; x++) {
				pt.x =  x * size;
				pt.y =  y * size;
				pt.z =  z * size;
				//cerr<<" pt="<<pt<<endl;

				if(! (distance(pt.x, pt.y, pt.z, _initData.center.x, _initData.center.y, _initData.center.z) < _initData.radius) ){
					continue; //such cell will not be inside spherical region
				}

				if (BoundaryStrategy::getInstance()->isValid(pt)){
					cell = potts->createCellG(pt);
					cell->type=initCellType(_initData);
					potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
					//It is necessary to do it this way because steppers are called only when we are performing pixel copies
					// but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
					//inventory unless you call steppers(VolumeTrackerPlugin) explicitely

					
				}
				else{
					continue;
				}

				for (cellPt.z = pt.z; cellPt.z < pt.z + cellWidth &&
					cellPt.z < dim.z; cellPt.z++)
					for (cellPt.y = pt.y; cellPt.y < pt.y + cellWidth &&
						cellPt.y < dim.y; cellPt.y++)
						for (cellPt.x = pt.x; cellPt.x < pt.x + cellWidth &&
							cellPt.x < dim.x; cellPt.x++){

								if (BoundaryStrategy::getInstance()->isValid(pt))
									cellField->set(cellPt, cell);

						}
						potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
						//It is necessary to do it this way because steppers are called only when we are performing pixel copies
						// but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
						//inventory unless you call steppers(VolumeTrackerPlugin) explicitely

			}



}

unsigned char BlobFieldInitializer::initCellType(const BlobFieldInitializerData & _initData){
	Automaton * automaton=potts->getAutomaton();
	if(_initData.typeNames.size()==0){//by default each newly created type will be 1 
		return 1;
	}/*else if (_initData.typeNames.size()==1){ //user specifie just one type
	 return automaton->getTypeId(_initData.typeNames[0]);
	 }*/else{ //user has specified more than one cell type - will pick randomly the type
		 BasicRandomNumberGenerator * randGen=BasicRandomNumberGenerator::getInstance();
		 int index = randGen->getInteger(0, _initData.typeNames.size()-1);        
		 return automaton->getTypeId(_initData.typeNames[index]);
	}

}

void BlobFieldInitializer::start() {
	if (sim->getRestartEnabled()){
		return ;  // we will not initialize cells if restart flag is on
	}
	// TODO: Chage this code so it write the 0 spins too.  This will make it
	//       possible to re-initialize a previously used field.

	/// I am changing here so that now I will work with cellFieldG - the field of CellG
	/// - this way CompuCell will have more functionality

	//  std::vector<BlobFieldInitializerData> & initDataVec=bipdPtr->initDataVec;
	//  int size = bipdPtr->gap + bipdPtr->width;

	WatchableField3D<CellG *> *cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
	ASSERT_OR_THROW("initField() Cell field G cannot be null!", cellFieldG);

	cerr<<"********************BLOB INIT***********************"<<endl;
	Dim3D dim = cellFieldG->getDim();
	if(blobInitializerData.size()!=0){
		for (int i = 0 ; i < blobInitializerData.size(); ++i){
			cerr<<"GOT HERE"<<endl;
			layOutCells(blobInitializerData[i]);
			//          exit(0);
		}
	}else{
		oldStyleInitData.center=Point3D(dim.x / 2,dim.y / 2,dim.z / 2);
		layOutCells(oldStyleInitData);

		if(cellSortInit){
			initializeCellTypesCellSort();
		}

		if(engulfmentData.engulfment){
			initializeEngulfment();
		}
	}




}

Dim3D BlobFieldInitializer::getBlobDimensions(const Dim3D & dim,int size){
	Dim3D itDim;

	itDim.x = dim.x / size;
	if (dim.x % size) itDim.x += 1;
	itDim.y = dim.y / size;
	if (dim.y % size) itDim.y += 1;
	itDim.z = dim.z / size;
	if (dim.z % size) itDim.z += 1;

	blobDim=itDim;

	return itDim; 

}


void BlobFieldInitializer::initializeEngulfment(){

	unsigned char topId,bottomId;
	CellTypePlugin * cellTypePluginPtr=(CellTypePlugin*)(Simulator::pluginManager.get("CellType"));
	ASSERT_OR_THROW("CellType plugin not initialized!", cellTypePluginPtr);

	EngulfmentData &enData=engulfmentData;

	topId=cellTypePluginPtr->getTypeId(enData.topType);
	bottomId=cellTypePluginPtr->getTypeId(enData.bottomType);

	cerr<<"topId="<<(int)topId<<" bottomId="<<(int)bottomId<<" enData.engulfmentCutoff="<<enData.engulfmentCutoff<<" enData.engulfmentCoordinate="<<enData.engulfmentCoordinate<<endl;




	WatchableField3D<CellG *> *cellFieldG =(WatchableField3D<CellG *> *) potts->getCellFieldG();
	Dim3D dim = cellFieldG->getDim();

	CellInventory * cellInventoryPtr=& potts->getCellInventory();
	///will initialize cell type to be 1
	CellInventory::cellInventoryIterator cInvItr;

	CellG * cell;
	Point3D pt;

	///loop over all the cells in the inventory   
	for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr ){
		cell=cellInventoryPtr->getCell(cInvItr);
		
		//cell=*cInvItr;
		cell->type=1; 
	}

	for (int x = 0 ; x < dim.x ; ++x){
		for (int y = 0 ; y < dim.y ; ++y){
			for (int z = 0 ; z < dim.z ; ++z){
				pt.x=x;
				pt.y=y;
				pt.z=z;
				cell=cellFieldG->get(pt);

				if(enData.engulfmentCoordinate=="x" || enData.engulfmentCoordinate=="X"){
					if(cell && pt.x<enData.engulfmentCutoff){
						cell->type=bottomId;
					}else if(cell && pt.x>=enData.engulfmentCutoff){
						cell->type=topId;
					}

				}
				if(enData.engulfmentCoordinate=="y" || enData.engulfmentCoordinate=="Y"){
					if(cell && pt.y<enData.engulfmentCutoff){
						cell->type=bottomId;
					}else if(cell && pt.y>=enData.engulfmentCutoff){
						cell->type=topId;
					}
				}
				if(enData.engulfmentCoordinate=="z" || enData.engulfmentCoordinate=="Z"){
					if(cell && pt.z<enData.engulfmentCutoff){
						cell->type=bottomId;
					}else if(cell && pt.z>=enData.engulfmentCutoff){
						cell->type=topId;
					}
				}

			}

		}
	}


}


void BlobFieldInitializer::initializeCellTypesCellSort(){
	//Note that because cells are ordered by physical address in the memory you get additional 
	//randomization of the cell types assignment. Assuming that randiom number generator is fixed i.e. it produces
	//same sequence of numbers every run, you still get random initial configuration and it comes from the fact that 
	// in general ordering of cells in the inventory is not repetitive between runs

	BasicRandomNumberGenerator *rand = BasicRandomNumberGenerator::getInstance();
	CellInventory * cellInventoryPtr=& potts->getCellInventory();

	///will initialize cell type here depending on the position of the cells
	CellInventory::cellInventoryIterator cInvItr;

	CellG * cell;

	///loop over all the cells in the inventory   
	for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr ){
		
		cell=cellInventoryPtr->getCell(cInvItr);
		//cell=*cInvItr;

		if(rand->getRatio()<0.5){ /// randomly assign types for cell sort
			cell->type=1;
		}else{
			cell->type=2;
		}

	}

}


//void BlobFieldInitializer::readXML(XMLPullParser &in) {
//  in.skip(TEXT);
//  
//   
//   string cellSortOpt;
//
//   pd=&bipd;
//   
//  while (in.check(START_ELEMENT)) {
//    if (in.getName() == "Region"){
//      cerr<<"Inside Region Definition"<<endl;
//      in.match(START_ELEMENT);
//      in.skip(TEXT);
//
//      BlobFieldInitializerData initData;
//
//      while (in.check(START_ELEMENT)){
//         
//         if (in.getName() == "Radius"){
//            initData.radius=BasicString::parseUInteger(in.matchSimple());;
//         }else if (in.getName() == "Center"){
//            initData.center.readXML(in);
//            in.matchSimple();
//         }else if (in.getName() == "Types"){
//            initData.typeNamesString=in.matchSimple();
//         }else if (in.getName() == "Width"){
//            initData.width = BasicString::parseUInteger(in.matchSimple());
//            cerr<<"width="<<initData.width <<endl;
//         }else if (in.getName() == "Gap"){
//            initData.gap = BasicString::parseUInteger(in.matchSimple());
//         }else {
//            throw BasicException(string("Unexpected element '") + in.getName() + 
//			   "'!", in.getLocation());
//         }
//         
//         in.skip(TEXT);
//      }
//      in.match(END_ELEMENT);
//      //Checking whether input values are sane
//      //apending to the init data vector
//      bipd.initDataVec.push_back(initData);
////       exit(0);
//    }
//    else if (in.getName() == "Gap") {
//      bipd.gap = BasicString::parseUInteger(in.matchSimple());
//
//    } else if (in.getName() == "Width") {
//      bipd.width = BasicString::parseUInteger(in.matchSimple());
//
//    } else if (in.getName() == "CellSortInit") {
//
//      bipd.cellSortOpt=in.matchSimple();
//      
//
//    }
//    else if(in.getName()=="Engulfment"){
//      bipd.engulfmentData.engulfment=true;
//
//      if(in.findAttribute("BottomType")>=0){
//            bipd.engulfmentData.bottomType=in.getAttribute("BottomType").value;
//      }
//      if(in.findAttribute("TopType")>=0){
//            bipd.engulfmentData.topType=in.getAttribute("TopType").value;
//      }
//
//      if(in.findAttribute("EngulfmentCutoff")>=0){
//            bipd.engulfmentData.engulfmentCutoff=BasicString::parseUInteger(in.getAttribute("EngulfmentCutoff").value);
//      }
//
//
//      if(in.findAttribute("EngulfmentCoordinate")>=0){
//            bipd.engulfmentData.engulfmentCoordinate=in.getAttribute("EngulfmentCoordinate").value;
//      }
//
//      in.matchSimple();
//    }
//    else if (in.getName() == "Radius") {
//      bipd.radius = BasicString::parseUInteger(in.matchSimple());
//    }
//    else {
//      throw BasicException(string("Unexpected element '") + in.getName() + 
//			   "'!", in.getLocation());
//    }
//    
//
//    in.skip(TEXT);
//  }
//}
//
//void BlobFieldInitializer::writeXML(XMLSerializer &out) {
//}
