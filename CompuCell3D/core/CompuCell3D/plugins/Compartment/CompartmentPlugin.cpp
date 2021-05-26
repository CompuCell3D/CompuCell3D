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

#include "CompartmentPlugin.h"

CompartmentPlugin::CompartmentPlugin() : potts(0), depth(1),weightDistance(false) {
}

CompartmentPlugin::~CompartmentPlugin() {
  
}

void CompartmentPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

  xmlData=_xmlData;
  potts=simulator->getPotts();
  potts->registerEnergyFunctionWithName(this,"ContactCompartment");
  simulator->registerSteerableObject(this);

}

void CompartmentPlugin::extraInit(Simulator *simulator){
	update(xmlData,true);
}

double CompartmentPlugin::changeEnergy(const Point3D &pt,
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
            energy -= contactEnergy(oldCell, nCell) / distance;
         }
         if(nCell!=newCell){
            energy += contactEnergy(newCell, nCell) / distance;
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
               else {
                  energy -= contactEnergy(oldCell, nCell);
               }
            }
            else{
               energy -= contactEnergy(oldCell, nCell);
           }
         }



         if(nCell!=newCell){
            if((newCell != 0) && (nCell != 0)) {
               if((newCell->clusterId) == (nCell->clusterId)) {
                  energy += internalEnergy(newCell,nCell);
               }
               else{
                  energy += contactEnergy(newCell, nCell);

               }
            }
            else{
               energy += contactEnergy(newCell, nCell);

            }
         }

   }
  }
  
  return energy;
}


double CompartmentPlugin::contactEnergy(const CellG *cell1, const CellG *cell2) {
//          cerr << "Contact Energy is: " << contactEnergyArray[cell1 ? cell1->type : 0][cell2? cell2->type : 0] << endl;
   return contactEnergyArray[cell1 ? cell1->type : 0][cell2? cell2->type : 0];
}

double CompartmentPlugin::internalEnergy(const CellG *cell1, const CellG *cell2) {
//       cerr << "Internal Energy is: " << internalEnergyArray[cell1 ? cell1->type : 0][cell2? cell2->type : 0] << endl;
   return internalEnergyArray[cell1 ? cell1->type : 0][cell2? cell2->type : 0];
}
void CompartmentPlugin::setContactCompartmentEnergy(const string typeName1,
				     const string typeName2,
				     const double energy) {
                    
  unsigned char type1 = automaton->getTypeId(typeName1);
  unsigned char type2 = automaton->getTypeId(typeName2);

  contactEnergyArray[type1][type2] = energy;
  contactEnergyArray[type2][type1] = energy;
}

void CompartmentPlugin::setInternalEnergy(const string typeName1,
				     const string typeName2,
				     const double energy) {

  unsigned char type1 = automaton->getTypeId(typeName1);
  unsigned char type2 = automaton->getTypeId(typeName2);
    
  internalEnergyArray[type1][type2] = energy;
  internalEnergyArray[type2][type1] = energy;
}
int CompartmentPlugin::getIndex(const int type1, const int type2) const {
  if (type1 < type2) return ((type1 + 1) | ((type2 + 1) << 16));
  else return ((type2 + 1) | ((type1 + 1) << 16));
}

void CompartmentPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

	contactEnergyArray.clear();
	internalEnergyArray.clear();

	automaton = potts->getAutomaton();
	ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton);

	set<unsigned char> cellTypesSet;
	set<unsigned char> cellInternalTypesSet;

	CC3DXMLElementList energyVec=_xmlData->getElements("Energy");

	for (int i = 0 ; i<energyVec.size(); ++i){

		setContactCompartmentEnergy(energyVec[i]->getAttribute("Type1"), energyVec[i]->getAttribute("Type2"), energyVec[i]->getDouble());

		//inserting all the types to the set (duplicate are automatically eleminated) to figure out max value of type Id
		cellTypesSet.insert(automaton->getTypeId(energyVec[i]->getAttribute("Type1")));
		cellTypesSet.insert(automaton->getTypeId(energyVec[i]->getAttribute("Type2")));

	}


	//figuring out maximum cell type id used in the xml
	CC3DXMLElementList energyInternalVec=_xmlData->getElements("InternalEnergy");
	for (int i = 0 ; i<energyInternalVec.size(); ++i){

		setInternalEnergy(energyInternalVec[i]->getAttribute("Type1"), energyInternalVec[i]->getAttribute("Type2"), energyInternalVec[i]->getDouble());

		//inserting all the types to the set (duplicate are automatically eleminated) to figure out max value of type Id
		cellInternalTypesSet.insert(automaton->getTypeId(energyInternalVec[i]->getAttribute("Type1")));
		cellInternalTypesSet.insert(automaton->getTypeId(energyInternalVec[i]->getAttribute("Type2")));

	}
	
	for(auto& i : cellInternalTypesSet){
		for(auto& j : cellInternalTypesSet){

			cerr<<"internal_energy["<<to_string(i)<<"]["<<to_string(j)<<"]="<<internalEnergyArray[i][j]<<endl;

		}
	}

	for(auto& i : cellTypesSet){
		for(auto& j : cellTypesSet){

			cerr<<"contact["<<to_string(i)<<"]["<<to_string(j)<<"]="<<contactEnergyArray[i][j]<<endl;

		}
	}
   
	//Here I initialize max neighbor index for direct acces to the list of neighbors 
   boundaryStrategy=BoundaryStrategy::getInstance();
   maxNeighborIndex=0;

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

std::string CompartmentPlugin::toString(){
   return "ContactCompartment";
}

std::string CompartmentPlugin::steerableName(){
   return toString();
}
