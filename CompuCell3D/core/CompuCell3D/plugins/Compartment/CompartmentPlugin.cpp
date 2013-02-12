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
 // // // #include <CompuCell3D/Field3D/Field3D.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>

// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Automaton/Automaton.h>
using namespace CompuCell3D;


// // // #include <BasicUtils/BasicString.h>
// // // #include <BasicUtils/BasicException.h>



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

//       while (true) {
//          n = fieldG->getNeighbor(pt, token, distance, false);
//          if (distance > depth) break;
// 
//          nCell = fieldG->get(n);
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
   //default behaviour  no energy weighting 
//       while (true) {
//          n = fieldG->getNeighbor(pt, token, distance, false);
//          if (distance > depth) break;
// 
//          nCell = fieldG->get(n);
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
                    
  char type1 = automaton->getTypeId(typeName1);
  char type2 = automaton->getTypeId(typeName2);
    
  int index = getIndex(type1, type2);

  contactEnergies_t::iterator it = contactEnergies.find(index); //return an iterator for the contact Energy
  ASSERT_OR_THROW(string("Contact energy for ") + typeName1 + " " + typeName2 +
		  " already set!", it == contactEnergies.end());

  contactEnergies[index] = energy;
}

void CompartmentPlugin::setInternalEnergy(const string typeName1,
				     const string typeName2,
				     const double energy) {
                    cerr<< "PING\n";
  char type1 = automaton->getTypeId(typeName1);
  char type2 = automaton->getTypeId(typeName2);
    
  int index = getIndex(type1, type2);

  contactEnergies_t::iterator it = internalEnergies.find(index); //return an iterator for the contact Energy
  ASSERT_OR_THROW(string("Internalenergy for ") + typeName1 + " " + typeName2 +
		  " already set!", it == internalEnergies.end());

  internalEnergies[index] = energy;
}
int CompartmentPlugin::getIndex(const int type1, const int type2) const {
  if (type1 < type2) return ((type1 + 1) | ((type2 + 1) << 16));
  else return ((type2 + 1) | ((type1 + 1) << 16));
}

void CompartmentPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

	if(potts->getDisplayUnitsFlag()){
		Unit contactEnergyUnit=potts->getEnergyUnit()/powerUnit(potts->getLengthUnit(),2);




		CC3DXMLElement * unitsElem=_xmlData->getFirstElement("Units"); 
		if (!unitsElem){ //add Units element
			unitsElem=_xmlData->attachElement("Units");
		}

		if(unitsElem->getFirstElement("EnergyUnit")){
			unitsElem->getFirstElement("EnergyUnit")->updateElementValue(contactEnergyUnit.toString());
		}else{
			CC3DXMLElement * energyUnitElem = unitsElem->attachElement("EnergyUnit",contactEnergyUnit.toString());
		}
		if(unitsElem->getFirstElement("InternalEnergyUnit")){
			unitsElem->getFirstElement("InternalEnergyUnit")->updateElementValue(contactEnergyUnit.toString());
		}else{
			CC3DXMLElement * internalEnergyUnitElem = unitsElem->attachElement("InternalEnergyUnit",contactEnergyUnit.toString());
		}

	}


	contactEnergies.clear();
	internalEnergies.clear();
	contactEnergyArray.clear();
	internalEnergyArray.clear();

	automaton = potts->getAutomaton();
	ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)

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

		//setInternalEnergy(energyInternalVec[i]->getAttribute("Type1"), energyInternalVec[i]->getAttribute("Type2"), energyInternalVec[i]->getDouble());

		//inserting all the types to the set (duplicate are automatically eleminated) to figure out max value of type Id
		cellInternalTypesSet.insert(automaton->getTypeId(energyInternalVec[i]->getAttribute("Type1")));
		cellInternalTypesSet.insert(automaton->getTypeId(energyInternalVec[i]->getAttribute("Type2")));

	}
	
	//Now that we know all the types used in the simulation we will find size of the contactEnergyArray
	vector<unsigned char> cellTypesVector(cellTypesSet.begin(),cellTypesSet.end());//coping set to the vector
	
	

	int size= * (max_element(cellTypesVector.begin(),cellTypesVector.end()));
	size+=1;//if max element is e.g. 5 then size has to be 6 for an array to be properly allocated

	int index ;
	contactEnergyArray.assign(size,vector<double>(size,0.0));
	

	//Begin Internal Vector allocation
	vector<unsigned char> cellInternalTypesVector(cellInternalTypesSet.begin(),cellInternalTypesSet.end());//coping set to the vector
	int i_size= * (max_element(cellInternalTypesVector.begin(),cellInternalTypesVector.end()));


	i_size+=1;

	int i_index ;
	internalEnergyArray.assign(size,vector<double>(size,0.0));
	cerr <<"i_size: " << i_size <<"  size: " << size << endl;
	
	//once the contact matrix has been allocated we check xml elements and put energy values into the matrix 
	set<pair<char,char> > typePairsSet;
	for (int i = 0 ; i<energyInternalVec.size(); ++i){
	  string typeName1 = energyInternalVec[i]->getAttribute("Type1");
	  string typeName2 = energyInternalVec[i]->getAttribute("Type2");
	  char type1 = automaton->getTypeId(typeName1);
	  char type2 = automaton->getTypeId(typeName2);
	  

	  if (typePairsSet.find(pair<char,char>(type1,type2))!=typePairsSet.end() ||typePairsSet.find(pair<char,char>(type2,type1))!=typePairsSet.end()){
	
		ASSERT_OR_THROW(string("InternalEnergy for ") + typeName1 + " " + typeName2 + " already set!", false);

	  }

	  typePairsSet.insert(pair<char,char>(type1,type2));
	  internalEnergyArray[type1][type2] = energyInternalVec[i]->getDouble();
	}

	//symmetrizing internal contact energy matrix
	for(int i = 1 ; i < size ; ++i) {
		for(int j = 1 ; j < size ; ++j){
			if (internalEnergyArray[i][j]!=internalEnergyArray[j][i]){
				if (internalEnergyArray[i][j]!=0.0){
					internalEnergyArray[j][i]=internalEnergyArray[i][j];
				}else{
					internalEnergyArray[i][j]=internalEnergyArray[j][i];
				}
			}
		}
	}	

	////End Internal vector allocation
	//for(int i = 1 ; i < i_size ; ++i) {
	//	for(int j = 1 ; j < i_size ; ++j){
	//		cerr<<"i="<<i<<" j="<<j<<" i_size="<<i_size<<endl;
	//		cerr<<"cellInternalTypesVector[i-1]="<<(int)cellInternalTypesVector[i-1]<<endl;
	//		cerr<<"cellInternalTypesVector[j-1]="<<(int)cellInternalTypesVector[j-1]<<endl;
	//		cerr<<"cellInternalTypesVector.size()="<<cellInternalTypesVector.size()<<endl;
	//		i_index = getIndex(cellInternalTypesVector[i-1],cellInternalTypesVector[j-1]);

	//		internalEnergyArray[i][j] = internalEnergies[i_index];

	//	}
	//}
	
	
	for(int i = 0 ; i < size ; ++i){
		for(int j = 0 ; j < size ; ++j){

			cerr<<"internal_energy["<<i<<"]["<<j<<"]="<<internalEnergyArray[i][j]<<endl;

		}
	}


	for(int i = 0 ; i < size ; ++i) {
		for(int j = 0 ; j < size ; ++j){

			index = getIndex(cellTypesVector[i],cellTypesVector[j]);

			contactEnergyArray[i][j] = contactEnergies[index];

		}
	}
	cerr<<"size="<<size<<endl;

	for(int i = 0 ; i < size ; ++i){
		for(int j = 0 ; j < size ; ++j){

			cerr<<"contact["<<i<<"]["<<j<<"]="<<contactEnergyArray[i][j]<<endl;

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


