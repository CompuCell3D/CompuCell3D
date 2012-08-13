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



#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <CompuCell3D/Boundary/BoundaryStrategy.h>
#include <CompuCell3D/Potts3D/CellInventory.h>
#include <CompuCell3D/Automaton/Automaton.h>
using namespace CompuCell3D;


#include <BasicUtils/BasicString.h>
#include <BasicUtils/BasicException.h>
#include <PublicUtilities/StringUtils.h>
#include <PublicUtilities/ParallelUtilsOpenMP.h>
#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>
#include <algorithm>



#include "ContactLocalProductPlugin.h"




ContactLocalProductPlugin::ContactLocalProductPlugin() : 
    pUtils(0),
	xmlData(0), 
   depth(1),
   weightDistance(false),
   contactEnergyPtr(&ContactLocalProductPlugin::contactEnergyLinear),
   maxNeighborIndex(0),
   boundaryStrategy(0),
   energyOffset(0.0)
   

{

}

ContactLocalProductPlugin::~ContactLocalProductPlugin() {
  
}

void ContactLocalProductPlugin::setJVecValue(CellG * _cell, unsigned int _index,float _value){
   (contactProductDataAccessor.get(_cell->extraAttribPtr)->jVec)[_index]=_value;
}

float ContactLocalProductPlugin::getJVecValue(CellG * _cell, unsigned int _index){
   return (contactProductDataAccessor.get(_cell->extraAttribPtr)->jVec)[_index];
}


void ContactLocalProductPlugin::setCadherinConcentration(CellG * _cell, unsigned int _index,float _value){
	setJVecValue(_cell,_index,_value);
}
float ContactLocalProductPlugin::getCadherinConcentration(CellG * _cell, unsigned int _index){
	return getJVecValue(_cell,_index);
}

void ContactLocalProductPlugin::setCadherinConcentrationVec(CellG * _cell, std::vector<float> &_vec){
	contactProductDataAccessor.get(_cell->extraAttribPtr)->jVec=_vec;
}

std::vector<float> ContactLocalProductPlugin::getCadherinConcentrationVec(CellG * _cell){
		
	return contactProductDataAccessor.get(_cell->extraAttribPtr)->jVec;
}


void ContactLocalProductPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
  
  xmlData=_xmlData;

  sim=simulator;
  potts=simulator->getPotts();
  
  pUtils=sim->getParallelUtils();
  
  potts->getCellFactoryGroupPtr()->registerClass(&contactProductDataAccessor);
  
  potts->registerEnergyFunctionWithName(this,"ContactLocalProduct");
  simulator->registerSteerableObject(this);

}

void ContactLocalProductPlugin::extraInit(Simulator *simulator){
	update(xmlData,true);
}



void ContactLocalProductPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

	automaton = potts->getAutomaton();
	ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)
		set<unsigned char> cellTypesSet;
	contactEnergies.clear();

	CC3DXMLElementList energyVec=_xmlData->getElements("ContactSpecificity");
																		 

	for (int i = 0 ; i<energyVec.size(); ++i){

		setContactEnergy(energyVec[i]->getAttribute("Type1"), energyVec[i]->getAttribute("Type2"), energyVec[i]->getDouble());

		//inserting all the types to the set (duplicate are automatically eleminated) to figure out max value of type Id
		cellTypesSet.insert(automaton->getTypeId(energyVec[i]->getAttribute("Type1")));
		cellTypesSet.insert(automaton->getTypeId(energyVec[i]->getAttribute("Type2")));

	}

	//Now that we know all the types used in the simulation we will find size of the contactEnergyArray
	vector<unsigned char> cellTypesVector(cellTypesSet.begin(),cellTypesSet.end());//coping set to the vector

	int size= * max_element(cellTypesVector.begin(),cellTypesVector.end());
	size+=1;//if max element is e.g. 5 then size has to be 6 for an array to be properly allocated

	int index ;
	contactSpecificityArray.clear();
	contactSpecificityArray.assign(size,vector<double>(size,0.0));
	

	for(int i = 0 ; i < size ; ++i)
		for(int j = 0 ; j < size ; ++j){

			index = getIndex(cellTypesVector[i],cellTypesVector[j]);

			contactSpecificityArray[i][j] = contactEnergies[index];

		}
		cerr<<"size="<<size<<endl;
		for(int i = 0 ; i < size ; ++i)
			for(int j = 0 ; j < size ; ++j){

				cerr<<"contact["<<i<<"]["<<j<<"]="<<contactSpecificityArray[i][j]<<endl;

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
			if(_xmlData->findElement("EnergyOffset")){
				energyOffset=_xmlData->getFirstElement("EnergyOffset")->getDouble();
			}
			if(_xmlData->findElement("ContactFunctionType")){
				contactFunctionType=_xmlData->getFirstElement("ContactFunctionType")->getText();
				changeToLower(contactFunctionType);

				if(contactFunctionType=="linear"){
				   if(_xmlData->getFirstElement("UseMediumLocal")){
						contactEnergyPtr=&ContactLocalProductPlugin::contactEnergyLinearMediumLocal;
				   }else{
						contactEnergyPtr=&ContactLocalProductPlugin::contactEnergyLinear;
				   }
					
				}else if(contactFunctionType=="quadratic"){
					if(_xmlData->getFirstElement("UseMediumLocal")){
						contactEnergyPtr=&ContactLocalProductPlugin::contactEnergyQuadraticMediumLocal;
					}else{
						contactEnergyPtr=&ContactLocalProductPlugin::contactEnergyQuadratic;
					}

				}else if (contactFunctionType=="min"){
					if(_xmlData->getFirstElement("UseMediumLocal")){
						contactEnergyPtr=&ContactLocalProductPlugin::contactEnergyMinMediumLocal;
					}else{
						contactEnergyPtr=&ContactLocalProductPlugin::contactEnergyMin;
					}
					
				}else{
					if(_xmlData->getFirstElement("UseMediumLocal")){
						contactEnergyPtr=&ContactLocalProductPlugin::contactEnergyLinearMediumLocal;
					}else{
						contactEnergyPtr=&ContactLocalProductPlugin::contactEnergyLinear;
					}
					
				}

			}

			if(_xmlData->findElement("CustomFunction")){
            
                //vectorized variables for convenient parallel access 
               unsigned int maxNumberOfWorkNodes=pUtils->getMaxNumberOfWorkNodesPotts();
               k1Vec.assign(maxNumberOfWorkNodes,0.0);
               k2Vec.assign(maxNumberOfWorkNodes,0.0);
               pVec.assign(maxNumberOfWorkNodes,mu::Parser());    

               // for (int i  = 0 ; i< maxNumberOfWorkNodes ; ++i){
                // pVec[i].DefineVar("Molecule1",&molecule1Vec[i]);
                // pVec[i].DefineVar("Molecule2",&molecule2Vec[i]);
                // pVec[i].SetExpr(formulaString);
               // }            
               
				// p=mu::Parser(); //using new parser
				CC3DXMLElementList variableVec=_xmlData->getFirstElement("CustomFunction")->getElements("Variable");
				int variableCount=0;
				bool variableInitializationOK=false;
				for (int i = 0 ; i<variableVec.size(); ++i){
					if (variableCount==0){
						cerr<<"ADDING VARIABLE "<<variableVec[i]->getText()<<endl;
						// p.DefineVar(variableVec[i]->getText(), &k1);
                        
                        for (int idx  = 0 ; idx< maxNumberOfWorkNodes ; ++idx){
                            pVec[idx].DefineVar(variableVec[i]->getText(), &k1Vec[idx]);
                        }
                        
					}
					else{
						cerr<<"ADDING VARIABLE "<<variableVec[i]->getText()<<endl;
						// p.DefineVar(variableVec[i]->getText(), &k2);					
                        for (int idx  = 0 ; idx< maxNumberOfWorkNodes ; ++idx){
                            pVec[idx].DefineVar(variableVec[i]->getText(), &k2Vec[idx]);
                        }
                        
					}
					++variableCount;
					if (variableCount==2){
						variableInitializationOK=true;
						break;
						
					}

				}


				ASSERT_OR_THROW("You need to list two variable names that will hold concentration of cadherins",variableInitializationOK);
				
				if(_xmlData->getFirstElement("CustomFunction")->findElement("Expression")){
					customExpression=_xmlData->getFirstElement("CustomFunction")->getFirstElement("Expression")->getText();
					cerr<<"THIS IS THE EXPRESSION="<<customExpression<<endl;
					// p.SetExpr(customExpression);
                    for (int idx  = 0 ; idx< maxNumberOfWorkNodes ; ++idx){
                        pVec[idx].SetExpr(customExpression);
                    }                    
					//uses different function depending on if cell-medium energy will be set for cell type or for each cell individually
					if(_xmlData->getFirstElement("UseMediumLocal")){
						contactEnergyPtr=&ContactLocalProductPlugin::contactEnergyCustomMediumLocal;
					}else{
						contactEnergyPtr=&ContactLocalProductPlugin::contactEnergyCustom;
					}
				}
			}

			cerr<<"Contact maxNeighborIndex="<<maxNeighborIndex<<endl;
			
}

double ContactLocalProductPlugin::changeEnergy(const Point3D &pt,
                                  const CellG *newCell,
                                  const CellG *oldCell) {
   //cerr<<"ChangeEnergy"<<endl;
   
   
  double energy = 0;
  unsigned int token = 0;
  double distance = 0;
//   Point3D n;
  Neighbor neighbor;
  
  CellG *nCell=0;
  WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();

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
			   if((nCell->clusterId) != (oldCell->clusterId)) {
				  energy -= (this->*contactEnergyPtr)(oldCell,nCell)/ neighbor.distance;
			   }
			}else{
			   energy -= (this->*contactEnergyPtr)(oldCell, nCell)/ neighbor.distance;
		   }
            
         }
         if(nCell!=newCell){
			if((newCell != 0) && (nCell != 0)) {
			   if((newCell->clusterId) != (nCell->clusterId)) {
				  energy += (this->*contactEnergyPtr)(newCell,nCell)/ neighbor.distance;
			   }
			}
			else{
			   energy += (this->*contactEnergyPtr)(newCell, nCell)/ neighbor.distance;

			}	            
         }
      }
  }else{
   //default behaviour  no energy weighting 
      for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
         neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
         if(!neighbor.distance){
         //if distance is 0 then the neighbor returned is invalid
         continue;
         }
         nCell = fieldG->get(neighbor.pt);

         if(nCell!=oldCell){
				if((nCell != 0) && (oldCell != 0)) {
				   if((nCell->clusterId) != (oldCell->clusterId)) {
					  energy -= (this->*contactEnergyPtr)(oldCell,nCell);
				   }
				}else{
				   energy -= (this->*contactEnergyPtr)(oldCell, nCell);
			   }            
         }
         if(nCell!=newCell){
			if((newCell != 0) && (nCell != 0)) {
			   if((newCell->clusterId) != (nCell->clusterId)) {
				  energy += (this->*contactEnergyPtr)(newCell,nCell);
			   }
			}
			else{
			   energy += (this->*contactEnergyPtr)(newCell, nCell);

			}            
         }
      }

   }

//   cerr<<"energy="<<energy<<endl;
  return energy;
}


double ContactLocalProductPlugin::contactEnergyLinear(const CellG *cell1, const CellG *cell2) {
   
   CellG *cell;
   CellG *neighbor;

   if(cell1){
      cell=const_cast<CellG *>(cell1);
      neighbor=const_cast<CellG *>(cell2);
   }else{
      cell=const_cast<CellG *>(cell2);
      neighbor=const_cast<CellG *>(cell1);
   }

   if(neighbor){
      vector<float> & jVecCell  = contactProductDataAccessor.get(cell->extraAttribPtr)->jVec;
      vector<float> & jVecNeighbor  = contactProductDataAccessor.get(neighbor->extraAttribPtr)->jVec;


      return energyOffset-jVecCell[0]*jVecNeighbor[0]*contactSpecificity(cell,neighbor);
   }else{
         return energyOffset- contactSpecificity(cell,neighbor);

   }

}

double ContactLocalProductPlugin::contactEnergyLinearMediumLocal(const CellG *cell1, const CellG *cell2) {
   
   CellG *cell;
   CellG *neighbor;

   if(cell1){
      cell=const_cast<CellG *>(cell1);
      neighbor=const_cast<CellG *>(cell2);
   }else{
      cell=const_cast<CellG *>(cell2);
      neighbor=const_cast<CellG *>(cell1);
   }

   if(neighbor){
      vector<float> & jVecCell  = contactProductDataAccessor.get(cell->extraAttribPtr)->jVec;
      vector<float> & jVecNeighbor  = contactProductDataAccessor.get(neighbor->extraAttribPtr)->jVec;


      return energyOffset-jVecCell[0]*jVecNeighbor[0]*contactSpecificity(cell,neighbor);
   }else{
	     vector<float> & jVecCell  = contactProductDataAccessor.get(cell->extraAttribPtr)->jVec;
         return energyOffset- jVecCell[1];

   }

}

double ContactLocalProductPlugin::contactEnergyQuadratic(const CellG *cell1, const CellG *cell2) {
   
   CellG *cell;
   CellG *neighbor;

   if(cell1){
      cell=const_cast<CellG *>(cell1);
      neighbor=const_cast<CellG *>(cell2);
   }else{
      cell=const_cast<CellG *>(cell2);
      neighbor=const_cast<CellG *>(cell1);
   }

   
   
   if(neighbor){
      vector<float> & jVecCell  = contactProductDataAccessor.get(cell->extraAttribPtr)->jVec;
      vector<float> & jVecNeighbor  = contactProductDataAccessor.get(neighbor->extraAttribPtr)->jVec;


      return energyOffset-jVecCell[0]*jVecCell[0]*jVecNeighbor[0]*jVecNeighbor[0]*contactSpecificity(cell,neighbor);
   }else{
         return energyOffset- contactSpecificity(cell,neighbor);

   }

}

double ContactLocalProductPlugin::contactEnergyQuadraticMediumLocal(const CellG *cell1, const CellG *cell2) {
   
   CellG *cell;
   CellG *neighbor;

   if(cell1){
      cell=const_cast<CellG *>(cell1);
      neighbor=const_cast<CellG *>(cell2);
   }else{
      cell=const_cast<CellG *>(cell2);
      neighbor=const_cast<CellG *>(cell1);
   }

   
   
   if(neighbor){
      vector<float> & jVecCell  = contactProductDataAccessor.get(cell->extraAttribPtr)->jVec;
      vector<float> & jVecNeighbor  = contactProductDataAccessor.get(neighbor->extraAttribPtr)->jVec;


      return energyOffset-jVecCell[0]*jVecCell[0]*jVecNeighbor[0]*jVecNeighbor[0]*contactSpecificity(cell,neighbor);
   }else{
	     vector<float> & jVecCell  = contactProductDataAccessor.get(cell->extraAttribPtr)->jVec;
         return energyOffset- jVecCell[1];

   }

}


double ContactLocalProductPlugin::contactEnergyMin(const CellG *cell1, const CellG *cell2) {
   
   CellG *cell;
   CellG *neighbor;

   if(cell1){
      cell=const_cast<CellG *>(cell1);
      neighbor=const_cast<CellG *>(cell2);
   }else{
      cell=const_cast<CellG *>(cell2);
      neighbor=const_cast<CellG *>(cell1);
   }

   
   
   
   if(neighbor){
      vector<float> & jVecCell  = contactProductDataAccessor.get(cell->extraAttribPtr)->jVec;
      vector<float> & jVecNeighbor  = contactProductDataAccessor.get(neighbor->extraAttribPtr)->jVec;


      return energyOffset-(jVecCell[0]<jVecNeighbor[0] ? jVecCell[0] : jVecNeighbor[0])*contactSpecificity(cell,neighbor);
   }else{
         return energyOffset- contactSpecificity(cell,neighbor);

   }

}

double ContactLocalProductPlugin::contactEnergyMinMediumLocal(const CellG *cell1, const CellG *cell2) {
   
   CellG *cell;
   CellG *neighbor;

   if(cell1){
      cell=const_cast<CellG *>(cell1);
      neighbor=const_cast<CellG *>(cell2);
   }else{
      cell=const_cast<CellG *>(cell2);
      neighbor=const_cast<CellG *>(cell1);
   }

   
   
   
   if(neighbor){
      vector<float> & jVecCell  = contactProductDataAccessor.get(cell->extraAttribPtr)->jVec;
      vector<float> & jVecNeighbor  = contactProductDataAccessor.get(neighbor->extraAttribPtr)->jVec;


      return energyOffset-(jVecCell[0]<jVecNeighbor[0] ? jVecCell[0] : jVecNeighbor[0])*contactSpecificity(cell,neighbor);
   }else{
	     vector<float> & jVecCell  = contactProductDataAccessor.get(cell->extraAttribPtr)->jVec;
         return energyOffset- jVecCell[1];

   }

}

double ContactLocalProductPlugin::contactEnergyCustom(const CellG *cell1, const CellG *cell2) {
   
   CellG *cell;
   CellG *neighbor;

   if(cell1){
      cell=const_cast<CellG *>(cell1);
      neighbor=const_cast<CellG *>(cell2);
   }else{
      cell=const_cast<CellG *>(cell2);
      neighbor=const_cast<CellG *>(cell1);
   }

   if(neighbor){
      vector<float> & jVecCell  = contactProductDataAccessor.get(cell->extraAttribPtr)->jVec;
      vector<float> & jVecNeighbor  = contactProductDataAccessor.get(neighbor->extraAttribPtr)->jVec;

        int currentWorkNodeNumber=pUtils->getCurrentWorkNodeNumber();	
        double & k1=k1Vec[currentWorkNodeNumber];
        double & k2=k2Vec[currentWorkNodeNumber];
        mu::Parser & p=pVec[currentWorkNodeNumber];
      
		k1=jVecCell[0];
		k2=jVecNeighbor[0];
		//cerr<<"k1="<<k1<<" k2="<<k2<<" result="<<p.Eval()<<endl;
		return energyOffset-p.Eval()*contactSpecificity(cell,neighbor);	
   }else{
         return energyOffset- contactSpecificity(cell,neighbor);

   }

}

double ContactLocalProductPlugin::contactEnergyCustomMediumLocal(const CellG *cell1, const CellG *cell2) {
   
   CellG *cell;
   CellG *neighbor;

   if(cell1){
      cell=const_cast<CellG *>(cell1);
      neighbor=const_cast<CellG *>(cell2);
   }else{
      cell=const_cast<CellG *>(cell2);
      neighbor=const_cast<CellG *>(cell1);
   }

   if(neighbor){
      vector<float> & jVecCell  = contactProductDataAccessor.get(cell->extraAttribPtr)->jVec;
      vector<float> & jVecNeighbor  = contactProductDataAccessor.get(neighbor->extraAttribPtr)->jVec;


        int currentWorkNodeNumber=pUtils->getCurrentWorkNodeNumber();	
        double & k1=k1Vec[currentWorkNodeNumber];
        double & k2=k2Vec[currentWorkNodeNumber];
        mu::Parser & p=pVec[currentWorkNodeNumber];
        
		k1=jVecCell[0];
		k2=jVecNeighbor[0];
		//cerr<<"k1="<<k1<<" k2="<<k2<<" result="<<p.Eval()<<endl;
		return energyOffset-p.Eval()*contactSpecificity(cell,neighbor);	
   }else{
	     vector<float> & jVecCell  = contactProductDataAccessor.get(cell->extraAttribPtr)->jVec;
         return energyOffset- jVecCell[1];

   }

}
double ContactLocalProductPlugin::contactSpecificity(const CellG *cell1, const CellG *cell2){
   return contactSpecificityArray[cell1 ? cell1->type : 0][cell2? cell2->type : 0];
}

void ContactLocalProductPlugin::setContactEnergy(const string typeName1,
				     const string typeName2,
				     const double energy) {
                    
  char type1 = automaton->getTypeId(typeName1);
  char type2 = automaton->getTypeId(typeName2);
    
  int index = getIndex(type1, type2);

  contactEnergies_t::iterator it = contactEnergies.find(index);
  ASSERT_OR_THROW(string("Contact energy for ") + typeName1 + " " + typeName2 +
		  " already set!", it == contactEnergies.end());

  contactEnergies[index] = energy;
}

int ContactLocalProductPlugin::getIndex(const int type1, const int type2) const {
  if (type1 < type2) return ((type1 + 1) | ((type2 + 1) << 16));
  else return ((type2 + 1) | ((type1 + 1) << 16));
}



std::string ContactLocalProductPlugin::steerableName(){
   return toString();
}

std::string ContactLocalProductPlugin::toString(){return "ContactLocalProduct";}

