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
#include "ContactInternalPlugin.h"


ContactInternalPlugin::ContactInternalPlugin() : potts(0), depth(1),weightDistance(false) {
}

ContactInternalPlugin::~ContactInternalPlugin() {
  
}

void ContactInternalPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

  xmlData=_xmlData;
  potts=simulator->getPotts();
  potts->registerEnergyFunctionWithName(this,"ContactInternal");
  simulator->registerSteerableObject(this);

}

void ContactInternalPlugin::extraInit(Simulator *simulator){
	update(xmlData,true);
}

double ContactInternalPlugin::changeEnergy(const Point3D &pt,
                                  const CellG *newCell,
                                  const CellG *oldCell) {
   //cerr<<"ChangeEnergy"<<endl;
   
   
  double energy = 0;
  unsigned int token = 0;
  double distance = 0;
  Point3D n;
  Neighbor neighbor;
  
  CellG *nCell=0;
  WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG*>*)potts->getCellFieldG();

   if(weightDistance){

      for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
         neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
         if(!neighbor.distance){
         //if distance is 0 then the neighbor returned is invalid
         continue;
         }
         nCell = fieldG->get(neighbor.pt);

         if(nCell!=oldCell){
            if((nCell != 0) && (oldCell != 0)) {
               if((nCell->clusterId) == (oldCell->clusterId)) {
                  energy -= internalEnergy(oldCell,nCell)/neighbor.distance;
               }
            }
            
         }
         if(nCell!=newCell){
            if((newCell != 0) && (nCell != 0)) {
               if((newCell->clusterId) == (nCell->clusterId)) {
                  energy += internalEnergy(newCell,nCell)/neighbor.distance;
               }
            }
         }
      }
  }else{

      for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
         neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
         if(!neighbor.distance){
         //if distance is 0 then the neighbor returned is invalid
         continue;
         }
         nCell = fieldG->get(neighbor.pt);

         if(nCell!=oldCell){
            if((nCell != 0) && (oldCell != 0)) {
               if((nCell->clusterId) == (oldCell->clusterId)) {
                  energy -= internalEnergy(oldCell,nCell);
               }
            }
         }

         if(nCell!=newCell){
            if((newCell != 0) && (nCell != 0)) {
               if((newCell->clusterId) == (nCell->clusterId)) {
                  energy += internalEnergy(newCell,nCell);
               }
            }
         }

   }
  }
  
  return energy;
}


double ContactInternalPlugin::internalEnergy(const CellG *cell1, const CellG *cell2) {
//       cerr << "Internal Energy is: " << internalEnergyArray[cell1 ? cell1->type : 0][cell2? cell2->type : 0] << endl;
   return internalEnergyArray[cell1 ? cell1->type : 0][cell2? cell2->type : 0];
}

void ContactInternalPlugin::setContactInternalEnergy(const string typeName1,
				     const string typeName2,
				     const double energy) {
  unsigned char type1 = automaton->getTypeId(typeName1);
  unsigned char type2 = automaton->getTypeId(typeName2);
    
  internalEnergyArray[type1][type2] = energy;
  internalEnergyArray[type2][type1] = energy;
}

void ContactInternalPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

	//if(potts->getDisplayUnitsFlag()){
	//	Unit contactEnergyUnit=potts->getEnergyUnit()/powerUnit(potts->getLengthUnit(),2);




	//	CC3DXMLElement * unitsElem=_xmlData->getFirstElement("Units"); 
	//	if (!unitsElem){ //add Units element
	//		unitsElem=_xmlData->attachElement("Units");
	//	}

	//	if(unitsElem->getFirstElement("EnergyUnit")){
	//		unitsElem->getFirstElement("EnergyUnit")->updateElementValue(contactEnergyUnit.toString());
	//	}else{
	//		CC3DXMLElement * energyUnitElem = unitsElem->attachElement("EnergyUnit",contactEnergyUnit.toString());
	//	}

	//}

	internalEnergyArray.clear();

	automaton = potts->getAutomaton();
	ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)

	set<unsigned char> cellTypesSet;
	

	CC3DXMLElementList energyVec=_xmlData->getElements("Energy");

	//figuring out maximum cell type id used in the xml
	for (int i = 0 ; i<energyVec.size(); ++i){

		setContactInternalEnergy(energyVec[i]->getAttribute("Type1"), energyVec[i]->getAttribute("Type2"), energyVec[i]->getDouble());

		//inserting all the types to the set (duplicate are automatically eleminated) to figure out max value of type Id
		cellTypesSet.insert(automaton->getTypeId(energyVec[i]->getAttribute("Type1")));
		cellTypesSet.insert(automaton->getTypeId(energyVec[i]->getAttribute("Type2")));

	}
	
	for(auto& i : cellTypesSet){
		for(auto& j : cellTypesSet){

			cerr<<"internal_energy["<<to_string(i)<<"]["<<to_string(j)<<"]="<<internalEnergyArray[i][j]<<endl;

		}
	}
   
	//Here I initialize max neighbor index for direct acces to the list of neighbors 
   boundaryStrategy=BoundaryStrategy::getInstance();
   maxNeighborIndex=0;

   if (_xmlData->getFirstElement("WeightEnergyByDistance")) {
       weightDistance = true;
   }

	if(_xmlData->getFirstElement("Depth")){
		maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromDepth(_xmlData->getFirstElement("Depth")->getDouble());
		//cerr<<"got here will do depth"<<endl;
	}else{
		//cerr<<"got here will do neighbor order"<<endl;
		if(_xmlData->getFirstElement("NeighborOrder")){

			maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(_xmlData->getFirstElement("NeighborOrder")->getUInt());	
		}else{
			maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);

		}

	}

	cerr<<"Contact maxNeighborIndex="<<maxNeighborIndex<<endl;

}

std::string ContactInternalPlugin::toString(){
   return "ContactInternal";
}

std::string ContactInternalPlugin::steerableName(){
   return toString();
}


