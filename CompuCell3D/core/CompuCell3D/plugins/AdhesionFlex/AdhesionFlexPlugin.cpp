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
// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>

// // // #include <CompuCell3D/Field3D/Field3D.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>
// // // #include <CompuCell3D/Boundary/BoundaryStrategy.h>

// // // #include <CompuCell3D/Potts3D/CellInventory.h>
// // // #include <CompuCell3D/Automaton/Automaton.h>
using namespace CompuCell3D;


// // // #include <BasicUtils/BasicString.h>
// // // #include <BasicUtils/BasicException.h>
// #include <PublicUtilities/StringUtils.h>
#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>
// // // #include <algorithm>


#include "AdhesionFlexPlugin.h"




AdhesionFlexPlugin::AdhesionFlexPlugin():
pUtils(0),
lockPtr(0),
xmlData(0) ,
numberOfAdhesionMolecules(0),
adhesionFlexEnergyPtr(&AdhesionFlexPlugin::adhesionFlexEnergyCustom),
weightDistance(false),
adhesionDensityInitialized(false)
{}

AdhesionFlexPlugin::~AdhesionFlexPlugin() {
	pUtils->destroyLock(lockPtr);
	delete lockPtr;
	lockPtr=0;
}

void AdhesionFlexPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
	xmlData=_xmlData;
	sim=simulator;
	potts=simulator->getPotts();
    
   pUtils=sim->getParallelUtils();
   lockPtr=new ParallelUtilsOpenMP::OpenMPLock_t;
   pUtils->initLock(lockPtr); 

   
   
	potts->getCellFactoryGroupPtr()->registerClass(&adhesionFlexDataAccessor);

	potts->registerEnergyFunctionWithName(this,"AdhesionFlex");
	simulator->registerSteerableObject(this);

}

void AdhesionFlexPlugin::extraInit(Simulator *simulator){
	update(xmlData,true);

}


void AdhesionFlexPlugin::handleEvent(CC3DEvent & _event){
	if (_event.id==CHANGE_NUMBER_OF_WORK_NODES){
    
        //vectorized variables for convenient parallel access 
       unsigned int maxNumberOfWorkNodes=pUtils->getMaxNumberOfWorkNodesPotts();
       molecule1Vec.assign(maxNumberOfWorkNodes,0.0);
       molecule2Vec.assign(maxNumberOfWorkNodes,0.0);
       pVec.assign(maxNumberOfWorkNodes,mu::Parser());    

       for (int i  = 0 ; i< maxNumberOfWorkNodes ; ++i){
        pVec[i].DefineVar("Molecule1",&molecule1Vec[i]);
        pVec[i].DefineVar("Molecule2",&molecule2Vec[i]);
        pVec[i].SetExpr(formulaString);
       }
    
	}
}


double AdhesionFlexPlugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {
	//cerr<<"ChangeEnergy"<<endl;
	if (!adhesionDensityInitialized){
        pUtils->setLock(lockPtr);        
		initializeAdhesionMoleculeDensityVector();		
        pUtils->unsetLock(lockPtr);        
	}
	//cerr<<"PROCESSING CHANGE OF ENERGY"<<endl;

	double energy = 0;
	unsigned int token = 0;
	double distance = 0;
	//   Point3D n;
	Neighbor neighbor;

	CellG *nCell=0;
	WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();

	//cerr<<"maxNeighborIndex="<<maxNeighborIndex<<endl;

	if(weightDistance){
		for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
			neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);

			if(!neighbor.distance){
				//if distance is 0 then the neighbor returned is invalid
				continue;
			}

			distance=neighbor.distance;

			nCell = fieldG->get(neighbor.pt);

			if(nCell!=oldCell){
				if((nCell != 0) && (oldCell != 0)) {
				   if((nCell->clusterId) != (oldCell->clusterId)) {
					  energy -= (this->*adhesionFlexEnergyPtr)(oldCell,nCell)/ neighbor.distance;
				   }
				}else{
				   energy -= (this->*adhesionFlexEnergyPtr)(oldCell, nCell)/ neighbor.distance;
			   }


			}
			if(nCell!=newCell){
				if((newCell != 0) && (nCell != 0)) {
				   if((newCell->clusterId) != (nCell->clusterId)) {
					  energy += (this->*adhesionFlexEnergyPtr)(newCell,nCell)/ neighbor.distance;
				   }
				}
				else{
				   energy += (this->*adhesionFlexEnergyPtr)(newCell, nCell)/ neighbor.distance;

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
			//cerr<<"nCell="<<nCell <<endl;
			//cerr<<" newCell="<<newCell<<endl;
			//cerr<<" oldCell="<<oldCell<<endl;
			if(nCell!=oldCell){
				if((nCell != 0) && (oldCell != 0)) {
				   if((nCell->clusterId) != (oldCell->clusterId)) {
					  energy -= (this->*adhesionFlexEnergyPtr)(oldCell,nCell);
				   }
				}else{
				   energy -= (this->*adhesionFlexEnergyPtr)(oldCell, nCell);
			   } 
				
			}
			if(nCell!=newCell){
			if((newCell != 0) && (nCell != 0)) {
			   if((newCell->clusterId) != (nCell->clusterId)) {
				  energy += (this->*adhesionFlexEnergyPtr)(newCell,nCell);
			   }
			}
			else{
			   energy += (this->*adhesionFlexEnergyPtr)(newCell, nCell);

			} 				
			}
		}

	}

	//cerr<<"energy="<<energy<<endl;
	return energy;
}


double AdhesionFlexPlugin::adhesionFlexEnergyCustom(const CellG *cell1, const CellG *cell2) {

	CellG *cell;
	CellG *neighbor;

	double energy=0.0;

	if(cell1){
		cell=const_cast<CellG *>(cell1);
		neighbor=const_cast<CellG *>(cell2);
	}else{
		cell=const_cast<CellG *>(cell2);
		neighbor=const_cast<CellG *>(cell1);
	}

    int currentWorkNodeNumber=pUtils->getCurrentWorkNodeNumber();	
    double & molecule1=molecule1Vec[currentWorkNodeNumber];
    double & molecule2=molecule2Vec[currentWorkNodeNumber];
    mu::Parser & p=pVec[currentWorkNodeNumber];

	if(neighbor){

		vector<float> & adhesionMoleculeDensityVecCell  = adhesionFlexDataAccessor.get(cell->extraAttribPtr)->adhesionMoleculeDensityVec;
		vector<float> & adhesionMoleculeDensityVecNeighbor  = adhesionFlexDataAccessor.get(neighbor->extraAttribPtr)->adhesionMoleculeDensityVec;
		//cerr<<"adhesionMoleculeDensityVecCell="<<adhesionMoleculeDensityVecCell.size()<<endl;
		//cerr<<"adhesionMoleculeDensityVecNeighbor="<<adhesionMoleculeDensityVecNeighbor.size()<<endl;

		//cerr<<"numberOfCadherins="<<numberOfCadherins<<endl;
		//cerr<<"cadherinSpecificityArray.size()="<<cadherinSpecificityArray.size()<<" "<<cadherinSpecificityArray[0].size()<<endl;
		//cerr<<"cell->type="<<(int)cell->type<<" neighbor->type="<<(int)neighbor->type<<endl;
		for (int i=0; i<numberOfAdhesionMolecules ; ++i)
			for (int j=0; j<numberOfAdhesionMolecules ; ++j){

				

				//cerr<<" i="<<i<<" j="<<j<<" adhesionMoleculeDensityVecCell[i]="<<adhesionMoleculeDensityVecCell[i]<<" adhesionMoleculeDensityVecNeighbor[j]"<<adhesionMoleculeDensityVecNeighbor[j]<<" cadherinSpecificityArray[i][j]="<<bindingParameterArray[i][j]<<endl;
				molecule1=adhesionMoleculeDensityVecCell[i];
				molecule2=adhesionMoleculeDensityVecNeighbor[j];
				energy-=p.Eval()*bindingParameterArray[i][j];

			}
			//cerr<<"energy after="<<energyOffset-energy<<endl;
			return energy;

	}else{
		//cerr<<"energy after contact with medium="<<-energy<<endl;

		vector<float> & adhesionMoleculeDensityVecCell  = adhesionFlexDataAccessor.get(cell->extraAttribPtr)->adhesionMoleculeDensityVec;
		vector<float> & adhesionMoleculeDensityVecNeighbor  = adhesionMoleculeDensityVecMedium;

		//cerr<<"1 adhesionMoleculeDensityVecCell="<<adhesionMoleculeDensityVecCell.size()<<endl;
		//cerr<<"1 adhesionMoleculeDensityVecNeighbor="<<adhesionMoleculeDensityVecNeighbor.size()<<endl;

		//cerr<<"cell->type="<<(int)cell->type<<" neighbor->type="<<0<<endl;	
		for (int i=0; i<numberOfAdhesionMolecules ; ++i)
			for (int j=0; j<numberOfAdhesionMolecules ; ++j){

				//cerr<<" i="<<i<<" j="<<j<<" jVecCell[i]="<<jVecCell[i]<<" jVecNeighbor[j]"<<jVecNeighbor[j]<<" cadherinSpecificityArray[i][j]="<<cadherinSpecificityArray[i][j]<<endl;
				molecule1=adhesionMoleculeDensityVecCell[i];
				molecule2=adhesionMoleculeDensityVecNeighbor[j];
				//cerr<<"p.Eval()="<<p.Eval()<<endl;
				energy-=p.Eval()*bindingParameterArray[i][j];


			}
			//cerr<<"energy after="<<energyOffset-energy<<endl;
			return energy;

	}

}



void AdhesionFlexPlugin::setBindingParameter(const std::string moleculeName1, const std::string moleculeName2, const double parameter, bool parsing_flag) {

	
	if (moleculeNameIndexMap.find(moleculeName1)==moleculeNameIndexMap.end()){
		cerr<<"CANNOT FIND MOLECULE 1 in the map"<<endl;
		ASSERT_OR_THROW(string("Molecule Name=")+moleculeName1+" was not declared in the AdhesionMolecule section",false);
	}

	if (moleculeNameIndexMap.find(moleculeName2)==moleculeNameIndexMap.end()){
		ASSERT_OR_THROW(string("Molecule Name=")+moleculeName2+" was not declared in the AdhesionMolecule section",false);
	}


	
	char molecule1 = moleculeNameIndexMap[moleculeName1];
	char molecule2 = moleculeNameIndexMap[moleculeName2];

	int index = getIndex(molecule1, molecule2);
	

	bindingParameters_t::iterator it = bindingParameters.find(index);
	
	if (parsing_flag){
		ASSERT_OR_THROW(string("BindingParameter for ") + moleculeName1 + " " + moleculeName2 +
			" already set!", it == bindingParameters.end());
	}
	

	bindingParameters[index] = parameter;
}


void AdhesionFlexPlugin::setBindingParameterDirect(const std::string moleculeName1, const std::string moleculeName2, const double parameter) {
	
	map<std::string, int>::iterator mitr_1=moleculeNameIndexMap.find(moleculeName1);
	map<std::string, int>::iterator mitr_2=moleculeNameIndexMap.find(moleculeName2);
	
	//if molecule name does not exist ignore it
	ASSERT_OR_THROW(string("setBindingParameterDirect: molecule name:") + moleculeName1 +string(" not found!"), mitr_1 !=moleculeNameIndexMap.end());
	ASSERT_OR_THROW(string("setBindingParameterDirect: molecule name:") + moleculeName2 +" not found!", mitr_2 !=moleculeNameIndexMap.end());
	
	
	bindingParameterArray[mitr_1->second][mitr_2->second] = parameter;
	bindingParameterArray[mitr_2->second][mitr_1->second] = parameter;	
}

void AdhesionFlexPlugin::setBindingParameterByIndexDirect(int _idx1, int _idx2, const double parameter) {
	bindingParameterArray[_idx1][_idx2] = parameter;
	bindingParameterArray[_idx2][_idx1] = parameter;
		
	
}


std::vector<std::vector<double> > AdhesionFlexPlugin::AdhesionFlexPlugin::getBindingParameterArray(){
	return bindingParameterArray;
}

std::vector<std::string> AdhesionFlexPlugin::getAdhesionMoleculeNameVec(){
	return 	adhesionMoleculeNameVec;
}


int AdhesionFlexPlugin::getIndex(const int type1, const int type2) const {
	if (type1 < type2) return ((type1 + 1) | ((type2 + 1) << 16));
	else return ((type2 + 1) | ((type1 + 1) << 16));
}


void AdhesionFlexPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

	automaton = potts->getAutomaton();
	ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)
		set<unsigned char> cellTypesSet;


	//scanning Adhesion Molecule names

	CC3DXMLElementList adhesionMoleculeNameXMLVec=_xmlData->getElements("AdhesionMolecule");
	
	set<string> adhesionMoleculeNameSet;
	adhesionMoleculeNameVec.clear();
	moleculeNameIndexMap.clear();

	for (int i = 0 ; i<adhesionMoleculeNameXMLVec.size(); ++i){
		string moleculeName=adhesionMoleculeNameXMLVec[i]->getAttribute("Molecule");
		if (! adhesionMoleculeNameSet.insert(moleculeName).second){
			ASSERT_OR_THROW(string("Duplicate molecule Name=")+moleculeName+ " specified in AdhesionMolecule section ", false);
		}        
		adhesionMoleculeNameVec.push_back(moleculeName);
		moleculeNameIndexMap.insert(make_pair(moleculeName,i));
	}

	numberOfAdhesionMolecules=moleculeNameIndexMap.size();
	if (!sim->getRestartEnabled()){	
		adhesionMoleculeDensityVecMedium=vector<float>(numberOfAdhesionMolecules,0.0);
	}
	cerr<<"numberOfAdhesionMolecules="<<numberOfAdhesionMolecules<<endl;

	//scannning AdhesionMoleculeDensity section

	CC3DXMLElementList adhesionMoleculeDensityXMLVec=_xmlData->getElements("AdhesionMoleculeDensity");    
	typeToAdhesionMoleculeDensityMap.clear();
	std::map<int,std::vector<float> >::iterator mitr;

	for (int i = 0 ; i<adhesionMoleculeDensityXMLVec.size(); ++i){
		int typeId=automaton->getTypeId(adhesionMoleculeDensityXMLVec[i]->getAttribute("CellType"));

		cerr<<"typeId="<<typeId<<endl;


		mitr=typeToAdhesionMoleculeDensityMap.find(typeId);
		if (mitr==typeToAdhesionMoleculeDensityMap.end()){
			typeToAdhesionMoleculeDensityMap.insert(make_pair(typeId,vector<float>(numberOfAdhesionMolecules,0.0)));
			cerr<<"typeToAdhesionMoleculeDensityMap[typeId].size()="<<typeToAdhesionMoleculeDensityMap[typeId].size()<<endl;
		}


		string moleculeName=adhesionMoleculeDensityXMLVec[i]->getAttribute("Molecule");
		cerr<<"moleculeName="<<moleculeName<<endl;
		if(moleculeNameIndexMap.find(moleculeName)==moleculeNameIndexMap.end()){
			ASSERT_OR_THROW(string("Molecule Name=")+moleculeName+" was not declared in the AdhesionMolecule section",false);
		}
		cerr<<"moleculeNameIndexMap[moleculeName]="<<moleculeNameIndexMap[moleculeName]<<endl;
		cerr<<"adhesionMoleculeDensityXMLVec[i]->getAttributeAsDouble(Density)="<<adhesionMoleculeDensityXMLVec[i]->getAttributeAsDouble("Density")<<endl;
		cerr<<"typeToAdhesionMoleculeDensityMap[typeId].size()="<<typeToAdhesionMoleculeDensityMap[typeId].size()<<endl;
		typeToAdhesionMoleculeDensityMap[typeId][moleculeNameIndexMap[moleculeName]]=adhesionMoleculeDensityXMLVec[i]->getAttributeAsDouble("Density");
		cerr<<"AFTER ASSIGNING DENSITY"<<endl;
	}


	//scanning BindingFormula section

	CC3DXMLElement * bindingFormulaXMLElem=_xmlData->getFirstElement("BindingFormula");

	formulaString=bindingFormulaXMLElem->getFirstElement("Formula")->getText(); //formula string

	CC3DXMLElement * variablesSectionXMLElem=bindingFormulaXMLElem->getFirstElement("Variables");
	//cerr<<"formulaString="<<formulaString<<endl;


	//here we can add options depending on variables input - for now it is har-coded to accept only matrix of bindingParameters
	bindingParameters.clear() ; //have to clear binding parameters
	CC3DXMLElement * adhesionInteractionMatrixXMLElem = variablesSectionXMLElem->getFirstElement("AdhesionInteractionMatrix");
	CC3DXMLElementList bindingParameterXMLVec = adhesionInteractionMatrixXMLElem->getElements("BindingParameter");
	for (int i = 0 ; i<bindingParameterXMLVec.size(); ++i){		
		setBindingParameter(bindingParameterXMLVec[i]->getAttribute("Molecule1") , bindingParameterXMLVec[i]->getAttribute("Molecule2"), bindingParameterXMLVec[i]->getDouble(),true);
	}


    //vectorized variables for convenient parallel access 
   unsigned int maxNumberOfWorkNodes=pUtils->getMaxNumberOfWorkNodesPotts();
   molecule1Vec.assign(maxNumberOfWorkNodes,0.0);
   molecule2Vec.assign(maxNumberOfWorkNodes,0.0);
   pVec.assign(maxNumberOfWorkNodes,mu::Parser());    

   for (int i  = 0 ; i< maxNumberOfWorkNodes ; ++i){
    pVec[i].DefineVar("Molecule1",&molecule1Vec[i]);
    pVec[i].DefineVar("Molecule2",&molecule2Vec[i]);
    pVec[i].SetExpr(formulaString);
   }

	// p=mu::Parser(); //using new parser
	// //setting up muParser
	// p.DefineVar("Molecule1", &molecule1);
	// p.DefineVar("Molecule2", &molecule2);	
	// p.SetExpr(formulaString);





	//scanning NeighborOrder or Depth
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
			cerr<<"maxNeighborIndex="<<maxNeighborIndex<<endl;
		}else{
			maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);

		}

	}


	cerr<<"sizeBindingArray="<<moleculeNameIndexMap.size()<<endl;
	//initializing binding parameter array
	int sizeBindingArray=moleculeNameIndexMap.size();

	int indexBindingArray ;
	bindingParameterArray.clear();
	bindingParameterArray.assign(sizeBindingArray,vector<double>(sizeBindingArray,0.0));
	bindingParameters_t::iterator bmitr;
	
	for(int i = 0 ; i < sizeBindingArray ; ++i)
		for(int j = 0 ; j < sizeBindingArray ; ++j){

			indexBindingArray= getIndex(i,j);

			bmitr=bindingParameters.find(indexBindingArray);

			if(bmitr!=bindingParameters.end()){
				bindingParameterArray[i][j] = bmitr->second;
			}


		}

		for(int i = 0 ; i < sizeBindingArray ; ++i)
			for(int j = 0 ; j < sizeBindingArray ; ++j){

				cerr<<"bindingParameterArray["<<i<<"]["<<j<<"]="<<bindingParameterArray[i][j]<<endl;

			}    

			


}


void AdhesionFlexPlugin::initializeAdhesionMoleculeDensityVector(){
	//cerr<<"initializeAdhesionMoleculeDensityVector adhesionDensityInitialized="<<adhesionDensityInitialized<<endl;
	//exit(1);

    if (adhesionDensityInitialized)//we double-check this flag to makes sure this function does not get called multiple times by different threads
        return;
    
	CellInventory::cellInventoryIterator cInvItr;
	CellG * cell;
	std::map<int,std::vector<float> >::iterator mitr;
	CellInventory * cellInventoryPtr=& potts->getCellInventory();


	for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr )
	{
		cell=cellInventoryPtr->getCell(cInvItr);		
		vector<float> & adhesionMoleculeDensityVecCell  = adhesionFlexDataAccessor.get(cell->extraAttribPtr)->adhesionMoleculeDensityVec;
		mitr=typeToAdhesionMoleculeDensityMap.find(cell->type);
		if (mitr!=typeToAdhesionMoleculeDensityMap.end()){
			adhesionMoleculeDensityVecCell=mitr->second;
		}else{
			adhesionMoleculeDensityVecCell=vector<float>(numberOfAdhesionMolecules,0.0);
		}

	}

	//initializing adhesionDensityVector for Medium	
	
	
	mitr=typeToAdhesionMoleculeDensityMap.find(automaton->getTypeId("Medium"));
	if (mitr!=typeToAdhesionMoleculeDensityMap.end()){
		adhesionMoleculeDensityVecMedium=mitr->second;
	}

    adhesionDensityInitialized=true;
}

void AdhesionFlexPlugin::setAdhesionMoleculeDensity(CellG * _cell, std::string _moleculeName, float _density){
	
	if (!_cell)	
		return;

	
	
	vector<float> & adhesionMoleculeDensityVec  = adhesionFlexDataAccessor.get(_cell->extraAttribPtr)->adhesionMoleculeDensityVec;
	map<std::string, int>::iterator mitr=moleculeNameIndexMap.find(_moleculeName);

	//if molecule name does not exist ignore it
	if(mitr!=moleculeNameIndexMap.end()){ 
		adhesionMoleculeDensityVec[mitr->second]=_density;	
	}
}



void AdhesionFlexPlugin::setAdhesionMoleculeDensityByIndex(CellG * _cell, int _idx, float _density){
	if (!_cell)
		return;
	
	vector<float> & adhesionMoleculeDensityVec  = adhesionFlexDataAccessor.get(_cell->extraAttribPtr)->adhesionMoleculeDensityVec;

	if (_idx>=adhesionMoleculeDensityVec.size() || _idx<0)
		return;

	adhesionMoleculeDensityVec[_idx]=_density;
}



void AdhesionFlexPlugin::setAdhesionMoleculeDensityVector(CellG * _cell, std::vector<float> _denVec){

	vector<float> & adhesionMoleculeDensityVec  = adhesionFlexDataAccessor.get(_cell->extraAttribPtr)->adhesionMoleculeDensityVec;
	//new vector will only be assigned if the sizes of the new and old vector match
	if (adhesionMoleculeDensityVec.size() == _denVec.size() ){
		adhesionMoleculeDensityVec=_denVec;
	}
}

void AdhesionFlexPlugin::assignNewAdhesionMoleculeDensityVector(CellG * _cell, std::vector<float> _denVec){

	vector<float> & adhesionMoleculeDensityVec  = adhesionFlexDataAccessor.get(_cell->extraAttribPtr)->adhesionMoleculeDensityVec;
	//assigns new vector regardless of the sizes of new and old vectors	
	adhesionMoleculeDensityVec=_denVec;
	
}



void AdhesionFlexPlugin::setMediumAdhesionMoleculeDensity(std::string _moleculeName, float _density){
	map<std::string, int>::iterator mitr=moleculeNameIndexMap.find(_moleculeName);
	//if molecule name does not exist ignore it
	if(mitr!=moleculeNameIndexMap.end()){ 
		adhesionMoleculeDensityVecMedium[mitr->second]=_density;	
	}
	
}
void AdhesionFlexPlugin::setMediumAdhesionMoleculeDensityByIndex(int _idx, float _density){
	if (_idx>=adhesionMoleculeDensityVecMedium.size() || _idx<0)
		return;

	adhesionMoleculeDensityVecMedium[_idx]=_density;
	
}

void AdhesionFlexPlugin::setMediumAdhesionMoleculeDensityVector(std::vector<float> _denVec){

	if(adhesionMoleculeDensityVecMedium.size()==_denVec.size())
		adhesionMoleculeDensityVecMedium=_denVec;
}

void AdhesionFlexPlugin::assignNewMediumAdhesionMoleculeDensityVector(std::vector<float> _denVec){

		//assigns new vector regardless of the sizes of new and old vectors	
		adhesionMoleculeDensityVecMedium=_denVec;

}



float AdhesionFlexPlugin::getAdhesionMoleculeDensity(CellG * _cell, std::string _moleculeName){
	if (!_cell)
		return errorDensity;

	vector<float> & adhesionMoleculeDensityVec  = adhesionFlexDataAccessor.get(_cell->extraAttribPtr)->adhesionMoleculeDensityVec;

	map<std::string, int>::iterator mitr=moleculeNameIndexMap.find(_moleculeName);

	//if molecule name does not exist ignore it
	if(mitr!=moleculeNameIndexMap.end()){ 
		return adhesionMoleculeDensityVec[mitr->second];	
	}
	return errorDensity;

}

float AdhesionFlexPlugin::getAdhesionMoleculeDensityByIndex(CellG * _cell, int _idx){
	if (!_cell)
		return errorDensity;

	vector<float> & adhesionMoleculeDensityVec  = adhesionFlexDataAccessor.get(_cell->extraAttribPtr)->adhesionMoleculeDensityVec;
	if (_idx>=adhesionMoleculeDensityVec.size() || _idx<0){
		return errorDensity;
	}
	return adhesionMoleculeDensityVec[_idx];
}

vector<float> AdhesionFlexPlugin::getAdhesionMoleculeDensityVector(CellG * _cell){
	if (!_cell)
		return vector<float>(1,errorDensity);
	
	vector<float> & adhesionMoleculeDensityVec  = adhesionFlexDataAccessor.get(_cell->extraAttribPtr)->adhesionMoleculeDensityVec;
	//cerr<<"ACCESSING adhesionMoleculeDensityVec size="<<adhesionMoleculeDensityVec.size()<<endl;
	return adhesionMoleculeDensityVec;
}


float AdhesionFlexPlugin::getMediumAdhesionMoleculeDensity(std::string _moleculeName){
	map<std::string, int>::iterator mitr=moleculeNameIndexMap.find(_moleculeName);
	//if molecule name does not exist ignore it
	if(mitr!=moleculeNameIndexMap.end()){ 
		return adhesionMoleculeDensityVecMedium[mitr->second];	
	}
	return errorDensity;
}

float AdhesionFlexPlugin::getMediumAdhesionMoleculeDensityByIndex(int _idx){
	if (_idx>=adhesionMoleculeDensityVecMedium.size() || _idx<0)
		return errorDensity;
	return adhesionMoleculeDensityVecMedium[_idx];
}
vector<float> AdhesionFlexPlugin::getMediumAdhesionMoleculeDensityVector(){

	return adhesionMoleculeDensityVecMedium;

}

void AdhesionFlexPlugin::overrideInitialization(){
	adhesionDensityInitialized=true;
	cerr<<"adhesionDensityInitialized="<<adhesionDensityInitialized<<endl;
}

std::string AdhesionFlexPlugin::toString(){
	return "AdhesionFlex";
}


std::string AdhesionFlexPlugin::steerableName(){
	return toString();
}


