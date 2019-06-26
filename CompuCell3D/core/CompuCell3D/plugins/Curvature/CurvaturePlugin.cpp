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
// // // #include <CompuCell3D/Potts3D/Cell.h>
// // // #include <CompuCell3D/Automaton/Automaton.h>
// // // #include <PublicUtilities/NumericalUtils.h>
// // // #include <PublicUtilities/ParallelUtilsOpenMP.h>
// // // #include <Utils/Coordinates3D.h>

using namespace CompuCell3D;


// // // #include <BasicUtils/BasicString.h>
// // // #include <BasicUtils/BasicException.h>
// // // #include <iostream>
// // // #include <algorithm>

using namespace std;

#include "CurvaturePlugin.h"




CurvaturePlugin::CurvaturePlugin():
pUtils(0),
xmlData(0)   
{
	lambda=0.0;
	activationEnergy=0.0;

	targetDistance=0.0;


	maxDistance=1000.0;



	diffEnergyFcnPtr=&CurvaturePlugin::diffEnergyByType;
	functionType=BYCELLTYPE;


}

CurvaturePlugin::~CurvaturePlugin() {

}




void CurvaturePlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
	potts=simulator->getPotts();
	fieldDim=potts->getCellFieldG()->getDim();


	xmlData=_xmlData;
	simulator->getPotts()->registerEnergyFunctionWithName(this,toString());
	simulator->registerSteerableObject(this);

	///will register FocalPointBoundaryPixelTracker here
	//BasicClassAccessorBase * cellFocalPointBoundaryPixelTrackerAccessorPtr=&CurvatureTrackerAccessor;
	///************************************************************************************************  
	///REMARK. HAVE TO USE THE SAME BASIC CLASS ACCESSOR INSTANCE THAT WAS USED TO REGISTER WITH FACTORY
	///************************************************************************************************  



	bool pluginAlreadyRegisteredFlag;
	Plugin *plugin=Simulator::pluginManager.get("CenterOfMass",&pluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded
	if(!pluginAlreadyRegisteredFlag)
		plugin->init(simulator);

	//first need to register center of mass plugin and then register Curvature
	potts->getCellFactoryGroupPtr()->registerClass(&curvatureTrackerAccessor);
	potts->registerCellGChangeWatcher(this);  

	pUtils=simulator->getParallelUtils();
	unsigned int maxNumberOfWorkNodes=pUtils->getMaxNumberOfWorkNodesPotts();        

	newJunctionInitiatedFlagWithinClusterVec.assign(maxNumberOfWorkNodes,false);
	newNeighborVec.assign(maxNumberOfWorkNodes,0);


}




void CurvaturePlugin::extraInit(Simulator *simulator){
	update(xmlData,true);
}


void CurvaturePlugin::handleEvent(CC3DEvent & _event){
    if (_event.id==CHANGE_NUMBER_OF_WORK_NODES){    
        unsigned int maxNumberOfWorkNodes=pUtils->getMaxNumberOfWorkNodesPotts();        

        newJunctionInitiatedFlagWithinClusterVec.assign(maxNumberOfWorkNodes,false);
        newNeighborVec.assign(maxNumberOfWorkNodes,0);
    }
    
    
}

void CurvaturePlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
	automaton = potts->getAutomaton();

	// set<unsigned char> cellTypesSet;
	set<unsigned char> internalCellTypesSet;
	// set<unsigned char> typeSpecCellTypesSet;
	set<unsigned char> internalTypeSpecCellTypesSet;


	// plastParams.clear();
	 internalCurvatureParams.clear();
	// typeSpecificCurvatureParams.clear();
	internalTypeSpecificCurvatureParams.clear();




	ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)

		//if(_xmlData->getFirstElement("Local")){
		//	diffEnergyFcnPtr=&CurvaturePlugin::diffEnergyLocal;
		//	return;
		//}



		// CC3DXMLElementList plastParamVec=_xmlData->getElements("Parameters");
		// if (plastParamVec.size()>0){
		// functionType=BYCELLTYPE;
		// }

		// for (int i = 0 ; i<plastParamVec.size(); ++i){

		// CurvatureTrackerData ctd;

		// char type1 = automaton->getTypeId(plastParamVec[i]->getAttribute("Type1"));
		// char type2 = automaton->getTypeId(plastParamVec[i]->getAttribute("Type2"));

		// int index = getIndex(type1, type2);

		// plastParams_t::iterator it = plastParams.find(index);
		// ASSERT_OR_THROW(string("Plasticity parameters for ") + type1 + " " + type2 +
		// " already set!", it == plastParams.end());


		// if(plastParamVec[i]->getFirstElement("Lambda"))
		// ctd.lambdaDistance=plastParamVec[i]->getFirstElement("Lambda")->getDouble();

		// if(plastParamVec[i]->getFirstElement("TargetDistance"))
		// ctd.targetDistance=plastParamVec[i]->getFirstElement("TargetDistance")->getDouble();

		// if(plastParamVec[i]->getFirstElement("ActivationEnergy")){
		// ctd.activationEnergy=plastParamVec[i]->getFirstElement("ActivationEnergy")->getDouble();
		// }

		// if(plastParamVec[i]->getFirstElement("MaxDistance"))
		// ctd.maxDistance=plastParamVec[i]->getFirstElement("MaxDistance")->getDouble();


		// plastParams[index] = ctd;



		// //inserting all the types to the set (duplicate are automatically eleminated) to figure out max value of type Id
		// cellTypesSet.insert(type1);
		// cellTypesSet.insert(type2);			

		// }


		// extracting internal parameters - used with compartmental cells
		CC3DXMLElementList internalCurvatureParamVec=_xmlData->getElements("InternalParameters");
	for (int i = 0 ; i<internalCurvatureParamVec.size(); ++i){

		CurvatureTrackerData ctd;

		char type1 = automaton->getTypeId(internalCurvatureParamVec[i]->getAttribute("Type1"));
		char type2 = automaton->getTypeId(internalCurvatureParamVec[i]->getAttribute("Type2"));

		int index = getIndex(type1, type2);
        cerr << "setting curvature parameters between type1=" << (int)type1 << " and type2=" << (int)type2 << endl;
		curvatureParams_t::iterator it = internalCurvatureParams.find(index);
		ASSERT_OR_THROW(string("Internal curvature parameters for ") + type1 + " " + type2 +
			" already set!", it == internalCurvatureParams.end());

		if(internalCurvatureParamVec[i]->getFirstElement("ActivationEnergy"))
			ctd.activationEnergy=internalCurvatureParamVec[i]->getFirstElement("ActivationEnergy")->getDouble();

		if(internalCurvatureParamVec[i]->getFirstElement("Lambda"))
			ctd.lambdaCurvature=internalCurvatureParamVec[i]->getFirstElement("Lambda")->getDouble();


		//if(internalCurvatureParamVec[i]->getFirstElement("MaxDistance"))
		//	ctd.maxDistance=internalCurvatureParamVec[i]->getFirstElement("MaxDistance")->getDouble();


		internalCurvatureParams[index] = ctd;



		//inserting all the types to the set (duplicate are automatically eleminated) to figure out max value of type Id
		internalCellTypesSet.insert(type1);
		internalCellTypesSet.insert(type2);			

	}


	// //extracting type specific parameters
	// CC3DXMLElement * typeSpecificParams=_xmlData->getFirstElement("TypeSpecificParameters");
	// CC3DXMLElementList typeSpecificCurvatureParamVec;
	// if(typeSpecificParams)
	// typeSpecificCurvatureParamVec=typeSpecificParams->getElements("Parameters");

	// for (int i = 0 ; i<typeSpecificCurvatureParamVec.size(); ++i){

	// CurvatureTrackerData ctd;

	// char type = automaton->getTypeId(typeSpecificCurvatureParamVec[i]->getAttribute("TypeName"));

	// if(typeSpecificCurvatureParamVec[i]->findAttribute("MaxNumberOfJunctions"))				
	// ctd.maxNumberOfJunctions=typeSpecificCurvatureParamVec[i]->getAttributeAsUInt("MaxNumberOfJunctions");

	// if(typeSpecificCurvatureParamVec[i]->findAttribute("NeighborOrder"))				
	// ctd.neighborOrder=typeSpecificCurvatureParamVec[i]->getAttributeAsUInt("NeighborOrder");

	// typeSpecificCurvatureParams[type]=ctd;
	// typeSpecCellTypesSet.insert(type);
	// }


	//extracting internal type specific parameters
	CC3DXMLElement * internalTypeSpecificParams=_xmlData->getFirstElement("InternalTypeSpecificParameters");
	CC3DXMLElementList internalTypeSpecificCurvatureParamVec;
	if(internalTypeSpecificParams)
		internalTypeSpecificCurvatureParamVec=internalTypeSpecificParams->getElements("Parameters");



	for (int i = 0 ; i<internalTypeSpecificCurvatureParamVec.size(); ++i){

		CurvatureTrackerData ctd;

		char type = automaton->getTypeId(internalTypeSpecificCurvatureParamVec[i]->getAttribute("TypeName"));

		if(internalTypeSpecificCurvatureParamVec[i]->findAttribute("MaxNumberOfJunctions"))				
			ctd.maxNumberOfJunctions=internalTypeSpecificCurvatureParamVec[i]->getAttributeAsUInt("MaxNumberOfJunctions");


		if(internalTypeSpecificCurvatureParamVec[i]->findAttribute("NeighborOrder"))				
			ctd.neighborOrder=internalTypeSpecificCurvatureParamVec[i]->getAttributeAsUInt("NeighborOrder");

		internalTypeSpecificCurvatureParams[type]=ctd;
		internalTypeSpecCellTypesSet.insert(type);		

	}

	// //Now that we know all the types used in the simulation we will find size of the plastParams
	// vector<unsigned char> cellTypesVector(cellTypesSet.begin(),cellTypesSet.end());//coping set to the vector

	// int size=0;
	// int index ;
	// if (cellTypesVector.size()){
	// size= * max_element(cellTypesVector.begin(),cellTypesVector.end());
	// size+=1;//if max element is e.g. 5 then size has to be 6 for an array to be properly allocated
	// }


	// plastParamsArray.clear();
	// plastParamsArray.assign(size,vector<CurvatureTrackerData>(size,CurvatureTrackerData()));



	// for(int i = 0 ; i < cellTypesVector.size() ; ++i)
	// for(int j = 0 ; j < cellTypesVector.size() ; ++j){
	// //cerr<<"cellTypesVector[i]="<<(int)cellTypesVector[i]<<endl;
	// //cerr<<"cellTypesVector[j]="<<(int)cellTypesVector[j]<<endl;
	// index = getIndex(cellTypesVector[i],cellTypesVector[j]);
	// //cerr<<"index="<<index <<endl;


	// plastParamsArray[cellTypesVector[i]][cellTypesVector[j]] = plastParams[index];

	// }


	//Now internal parameters
	//Now that we know all the types used in the simulation we will find size of the plastParams
	vector<unsigned char> internalCellTypesVector(internalCellTypesSet.begin(),internalCellTypesSet.end());//coping set to the vector

	int size=0;
	if (internalCellTypesVector.size()){
		size= * max_element(internalCellTypesVector.begin(),internalCellTypesVector.end());
		size+=1;//if max element is e.g. 5 then size has to be 6 for an array to be properly allocated
	}

	internalCurvatureParamsArray.clear();
	internalCurvatureParamsArray.assign(size,vector<CurvatureTrackerData>(size,CurvatureTrackerData()));
	int index;
	for(int i = 0 ; i < internalCellTypesVector.size() ; ++i)
		for(int j = 0 ; j < internalCellTypesVector.size() ; ++j){
			index = getIndex(internalCellTypesVector[i],internalCellTypesVector[j]);
			internalCurvatureParamsArray[internalCellTypesVector[i]][internalCellTypesVector[j]] = internalCurvatureParams[index];				
		}





		// //Now type specific parameters
		// //Now that we know all the types used in the simulation we will find size of the plastParams
		// vector<unsigned char> typeSpecCellTypesVector(typeSpecCellTypesSet.begin(),typeSpecCellTypesSet.end());//coping set to the vector

		// size=0;
		// if (typeSpecCellTypesVector.size()){
		// size= * max_element(typeSpecCellTypesVector.begin(),typeSpecCellTypesVector.end());
		// size+=1;//if max element is e.g. 5 then size has to be 6 for an array to be properly allocated
		// }

		// typeSpecificCurvatureParamsVec.clear();
		// typeSpecificCurvatureParamsVec.assign(size,CurvatureTrackerData());

		// for(int i = 0 ; i < typeSpecCellTypesVector.size() ; ++i){


		// typeSpecificCurvatureParamsVec[typeSpecCellTypesVector[i]]=typeSpecificCurvatureParams[typeSpecCellTypesVector[i]];


		// }


		// ASSERT_OR_THROW("THE NUMBER TYPE NAMES IN THE TYPE SPECIFIC SECTION DOES NOT MATCH THE NUMBER OF CELL TYPES IN PARAMETERS SECTION",typeSpecificCurvatureParamsVec.size()==plastParamsArray.size());
		//Now internal type specific parameters

		//Now that we know all the types used in the simulation we will find size of the plastParams
		vector<unsigned char> internalTypeSpecCellTypesVector(internalTypeSpecCellTypesSet.begin(),internalTypeSpecCellTypesSet.end());//coping set to the vector
		size=0;

		if(internalTypeSpecCellTypesVector.size()){ 
			size= * max_element(internalTypeSpecCellTypesVector.begin(),internalTypeSpecCellTypesVector.end());
			size+=1;//if max element is e.g. 5 then size has to be 6 for an array to be properly allocated
		}

		internalTypeSpecificCurvatureParamsVec.clear();
		internalTypeSpecificCurvatureParamsVec.assign(size,CurvatureTrackerData());

		for(int i = 0 ; i < internalTypeSpecCellTypesVector.size() ; ++i){


			internalTypeSpecificCurvatureParamsVec[internalTypeSpecCellTypesVector[i]]=internalTypeSpecificCurvatureParams[internalTypeSpecCellTypesVector[i]];						

		}

		ASSERT_OR_THROW("THE NUMBER TYPE NAMES IN THE INTERNAL TYPE SPECIFIC SECTION DOES NOT MATCH THE NUMBER OF CELL TYPES IN INTERNAL PARAMETERS SECTION",internalTypeSpecificCurvatureParamsVec.size()==internalCurvatureParamsArray.size());

		boundaryStrategy=BoundaryStrategy::getInstance();
}




double CurvaturePlugin::potentialFunction(double _lambda,double _offset, double _targetDistance , double _distance){
	return _offset+_lambda*(_distance-_targetDistance)*(_distance-_targetDistance);
}

//not used
double CurvaturePlugin::diffEnergyLocal(float _deltaL,float _lAfter,const CurvatureTrackerData * _curvatureTrackerData,const CellG *_cell){

	// float lambdaLocal=_curvatureTrackerData->lambdaDistance;
	// float targetDistanceLocal=_curvatureTrackerData->targetDistance;

	// if(_cell->volume>1){
	// return lambdaLocal*_deltaL*(2*(_lAfter-targetDistanceLocal)+_deltaL);
	// }else{//after spin flip oldCell will disappear so the only contribution from before spin flip i.e. -(l-l0)^2
	// return -lambdaLocal*(_lAfter-targetDistanceLocal)*(_lAfter-targetDistanceLocal);
	// }
	return 0.0;
}

//not used
double CurvaturePlugin::diffEnergyGlobal(float _deltaL,float _lAfter,const CurvatureTrackerData * _curvatureTrackerData,const CellG *_cell){

	// if(_cell->volume>1){
	// return lambda*_deltaL*(2*(_lAfter-targetDistance)+_deltaL);
	// }else{//after spin flip oldCell will disappear so the only contribution from before spin flip i.e. -(l-l0)^2
	// return -lambda*(_lAfter-targetDistance)*(_lAfter-targetDistance);
	// }
	return 0.0;
}


double CurvaturePlugin::diffEnergyByType(float _deltaL,float _lAfter,const CurvatureTrackerData * _curvatureTrackerData,const CellG *_cell){

	// if(_cell->volume>1){
	// double diffEnergy=_curvatureTrackerData->lambdaCurvature*_deltaL*(2*(_lAfter-_curvatureTrackerData->targetDistance)+_deltaL);
	// //cerr<<"_curvatureTrackerData->targetDistance="<<_curvatureTrackerData->targetDistance<<endl;
	// //cerr<<"_curvatureTrackerData->lambdaDistance="<<_curvatureTrackerData->lambdaDistance<<endl;
	// //cerr<<"diff energy="<<diffEnergy<<endl;
	// return diffEnergy;
	// }else{//after spin flip oldCell will disappear so the only contribution from before spin flip i.e. -(l-l0)^2
	// return -_curvatureTrackerData->lambdaDistance*(_lAfter-_curvatureTrackerData->targetDistance)*(_lAfter-_curvatureTrackerData->targetDistance);
	// }
	return 0.0;
}


// double CurvaturePlugin::tryAddingNewJunction(const Point3D &pt,const CellG *newCell) {
// //cerr<<"typeSpecificCurvatureParamsVec.size()="<<typeSpecificCurvatureParamsVec.size()<<endl;

// if (((int)typeSpecificCurvatureParamsVec.size())-1<newCell->type){ //the newCell type is not listed by the user
// newJunctionInitiatedFlag=false;
// return 0.0;
// }

// //check if new cell can accept new junctions
// if(CurvatureTrackerAccessor.get(newCell->extraAttribPtr)->CurvatureNeighbors.size()>=typeSpecificCurvatureParamsVec[newCell->type].maxNumberOfJunctions){
// newJunctionInitiatedFlag=false;
// return 0.0;

// }


// boundaryStrategy=BoundaryStrategy::getInstance();
// int maxNeighborIndexLocal=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(typeSpecificCurvatureParamsVec[newCell->type].neighborOrder);
// Neighbor neighbor;
// CellG * nCell;
// WatchableField3D<CellG *> *fieldG =(WatchableField3D<CellG *> *) potts->getCellFieldG();
// //visit point neighbors of newCell and see if within of specified range there is another cell with which newCell can make a junction

// for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndexLocal ; ++nIdx ){
// neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
// if(!neighbor.distance){
// //if distance is 0 then the neighbor returned is invalid
// continue;
// }
// nCell=fieldG->get(neighbor.pt);

// if (!nCell) //no junctions with medium
// continue;


// if (nCell==newCell || nCell->clusterId==newCell->clusterId)	// make sure that newCell and nCell are different and belong to different clusters
// continue;



// //check if type of neighbor cell is listed by the user
// if(((int)typeSpecificCurvatureParamsVec.size())-1<nCell->type || typeSpecificCurvatureParamsVec[nCell->type].maxNumberOfJunctions==0){	

// continue;
// }

// // check if neighbor cell can accept another junction
// if(CurvatureTrackerAccessor.get(nCell->extraAttribPtr)->CurvatureNeighbors.size()>=typeSpecificCurvatureParamsVec[nCell->type].maxNumberOfJunctions){
// //checkIfJunctionPossible=false;
// continue;
// }

// // check if new cell can accept another junction
// if(CurvatureTrackerAccessor.get(newCell->extraAttribPtr)->CurvatureNeighbors.size()>=typeSpecificCurvatureParamsVec[newCell->type].maxNumberOfJunctions){
// //checkIfJunctionPossible=false;
// continue;
// }


// //check if nCell has has a junction with newCell                
// set<CurvatureTrackerData>::iterator sitr=
// CurvatureTrackerAccessor.get(newCell->extraAttribPtr)->CurvatureNeighbors.find(CurvatureTrackerData(nCell));
// if(sitr==CurvatureTrackerAccessor.get(newCell->extraAttribPtr)->CurvatureNeighbors.end()){
// //new connection allowed
// newJunctionInitiatedFlag=true;
// newNeighbor=nCell;
// break; 

// }


// }


// if(newJunctionInitiatedFlag){
// //cerr<<"newCell->type="<<(int)newCell->type<<" newNeighbor->type="<<(int)newNeighbor->type<<" energy="<<plastParamsArray[newCell->type][newNeighbor->type].activationEnergy<<endl; 		
// return plastParamsArray[newCell->type][newNeighbor->type].activationEnergy;
// }	else{
// return 0.0;

// }

// }

double CurvaturePlugin::tryAddingNewJunctionWithinCluster(const Point3D &pt,const CellG *newCell) {
	//cerr<<"internalTypeSpecificCurvatureParamsVec.size()="<<internalTypeSpecificCurvatureParamsVec.size()<<endl;
	//cerr<<"newCell->type="<<(int)newCell->type<<endl;

	int currentWorkNodeNumber=pUtils->getCurrentWorkNodeNumber();	
	short &  newJunctionInitiatedFlagWithinCluster = newJunctionInitiatedFlagWithinClusterVec[currentWorkNodeNumber];
	CellG * & newNeighbor=newNeighborVec[currentWorkNodeNumber];

	if (((int)internalTypeSpecificCurvatureParamsVec.size())-1<newCell->type){ //the newCell type is not listed by the user
		newJunctionInitiatedFlagWithinCluster=false;
		return 0.0;
	}



	//check if new cell can accept new junctions
	if(curvatureTrackerAccessor.get(newCell->extraAttribPtr)->internalCurvatureNeighbors.size()>=internalTypeSpecificCurvatureParamsVec[newCell->type].maxNumberOfJunctions){
		newJunctionInitiatedFlagWithinCluster=false;
		return 0.0;

	}



	boundaryStrategy=BoundaryStrategy::getInstance();
	int maxNeighborIndexLocal=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(internalTypeSpecificCurvatureParamsVec[newCell->type].neighborOrder);
	Neighbor neighbor;
	CellG * nCell;
	WatchableField3D<CellG *> *fieldG =(WatchableField3D<CellG *> *) potts->getCellFieldG();
	//visit point neighbors of newCell and see if within of specified range there is another cell with which newCell can make a junction

	for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndexLocal ; ++nIdx ){
		neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
		if(!neighbor.distance){
			//if distance is 0 then the neighbor returned is invalid
			continue;
		}
		nCell=fieldG->get(neighbor.pt);

		if (!nCell) //no junctions with medium
			continue;


		if (nCell==newCell || nCell->clusterId!=newCell->clusterId)	// make sure that newCell and nCell are different and belong to the same clusters
			continue;



		//check if type of neighbor cell is listed by the user
		if(((int)internalTypeSpecificCurvatureParamsVec.size())-1<nCell->type || internalTypeSpecificCurvatureParamsVec[nCell->type].maxNumberOfJunctions==0){	

			continue;
		}

		// check if neighbor cell can accept another junction
		if(curvatureTrackerAccessor.get(nCell->extraAttribPtr)->internalCurvatureNeighbors.size()>=internalTypeSpecificCurvatureParamsVec[nCell->type].maxNumberOfJunctions){
			//checkIfJunctionPossible=false;
			continue;
		}
		// check if newCell can accept another junction
		if(curvatureTrackerAccessor.get(newCell->extraAttribPtr)->internalCurvatureNeighbors.size()>=internalTypeSpecificCurvatureParamsVec[newCell->type].maxNumberOfJunctions){
			//checkIfJunctionPossible=false;
			continue;
		}

		//check if nCell has has a junction with newCell                
		set<CurvatureTrackerData>::iterator sitr=
			curvatureTrackerAccessor.get(newCell->extraAttribPtr)->internalCurvatureNeighbors.find(CurvatureTrackerData(nCell));
		if(sitr==curvatureTrackerAccessor.get(newCell->extraAttribPtr)->internalCurvatureNeighbors.end()){
			//new connection allowed
			newJunctionInitiatedFlagWithinCluster=true;
			newNeighbor=nCell;
			//cerr<<"newNeighbor="<<newNeighbor<<endl;

			break; 

		}


	}


	if(newJunctionInitiatedFlagWithinCluster){
		//cerr<<"\t\t\t newCell="<<newCell<<" newNeighbor="<<newNeighbor<<endl;
		//cerr<<"\t\t\t newCell->type="<<(int)newCell->type<<" newNeighbor->type="<<(int)newNeighbor->type<<" energy="<<internalCurvatureParamsArray[newCell->type][newNeighbor->type].activationEnergy<<endl; 		
		//cerr<<"\t\t\t internal nCell->.maxNumberOfJunctions="<<internalTypeSpecificCurvatureParamsVec[nCell->type].maxNumberOfJunctions<<endl;
		//cerr<<"\t\t\t newCell number of internal neighbors "<<CurvatureTrackerAccessor.get(newCell->extraAttribPtr)->internalCurvatureNeighbors.size()<<endl;
		//cerr<<"\t\t\t internal newCell.maxNumberOfJunctions="<<internalTypeSpecificCurvatureParamsVec[newCell->type].maxNumberOfJunctions<<endl;
		//cerr<<"internalCurvatureParamsArray="<<internalCurvatureParamsArray.size()<<" internalCurvatureParamsArray.size().size()="<<internalCurvatureParamsArray[0].size()<<endl;
		//cerr<<"newCell->type="<<(int)newCell->type<<endl;
		//cerr<<"1 newNeighbor="<<newNeighbor<<endl;
		//cerr<<"newNeighbor->type="<<(int)newNeighbor->type<<endl;
		return internalCurvatureParamsArray[newCell->type][newNeighbor->type].activationEnergy;

	}	else{
		return 0.0;

	}

}

double CurvaturePlugin::calculateInverseCurvatureSquare(const Vector3 & _leftVec, const Vector3 & _middleVec , const Vector3 & _rightVec){
	Vector3 segmentVec=(_rightVec-_leftVec);	
	double segmentMagnitude=segmentVec.Mag();
	Vector3 leftMid=_leftVec-_middleVec;
	Vector3 rightMid=_rightVec-_middleVec;
	double sinLeftMidRight=fabs((float)leftMid.Cross(rightMid).Mag()/(leftMid.Mag()*rightMid.Mag()));
	return 2*sinLeftMidRight/segmentMagnitude;

}

double CurvaturePlugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {

	//This plugin will not work properly with periodic boundary conditions. If necessary I can fix it

	if (newCell==oldCell) //this may happen if you are trying to assign same cell to one pixel twice 
		return 0.0;


	double energy=0.0;
	WatchableField3D<CellG *> *fieldG =(WatchableField3D<CellG *> *) potts->getCellFieldG();

	Neighbor neighbor;
	Neighbor neighborOfNeighbor;
	CellG * nCell;
	CellG * nnCell;



	int currentWorkNodeNumber=pUtils->getCurrentWorkNodeNumber();	    
	short &  newJunctionInitiatedFlagWithinCluster = newJunctionInitiatedFlagWithinClusterVec[currentWorkNodeNumber];	
	CellG * & newNeighbor=newNeighborVec[currentWorkNodeNumber];

	newJunctionInitiatedFlagWithinCluster=false;
	newNeighbor=0;

	//check if we need to create new junctions only new cell can initiate junctions
	//cerr<<" ENERGY="<<energy<<endl;
	if(newCell){

		double activationEnergy=tryAddingNewJunctionWithinCluster(pt,newCell);
		if(newJunctionInitiatedFlagWithinCluster){
			//cerr<<"GOT NEW JUNCTION"<<endl;

			//exit(0);
			energy+=activationEnergy;

			return energy;
		}
	}


	// if(newCell){
	// double activationEnergy=tryAddingNewJunction(pt,newCell);		
	// if(newJunctionInitiatedFlag){
	// //cerr<<"GOT NEW JUNCTION"<<endl;

	// //exit(0);
	// energy+=activationEnergy;

	// return energy;
	// }
	// }



	Coordinates3D<double> centroidOldAfter;
	Coordinates3D<double> centroidNewAfter;
	Coordinates3D<float> centMassOldAfter;
	Coordinates3D<float> centMassNewAfter;



	//    cerr<<"fieldDim="<<fieldDim<<endl;
	if(oldCell){
		centMassOldAfter.XRef()=oldCell->xCM/(float)oldCell->volume;
		centMassOldAfter.YRef()=oldCell->yCM/(float)oldCell->volume;
		centMassOldAfter.ZRef()=oldCell->zCM/(float)oldCell->volume;

		if(oldCell->volume>1){
			centroidOldAfter=precalculateCentroid(pt, oldCell, -1,fieldDim,boundaryStrategy);
			centMassOldAfter.XRef()=centroidOldAfter.X()/(float)(oldCell->volume-1);
			centMassOldAfter.YRef()=centroidOldAfter.Y()/(float)(oldCell->volume-1);
			centMassOldAfter.ZRef()=centroidOldAfter.Z()/(float)(oldCell->volume-1);

		}else{
			//          return 0.0;//if oldCell is to disappear the Plasticity energy will be zero
			centroidOldAfter.XRef()=oldCell->xCM;
			centroidOldAfter.YRef()=oldCell->yCM;
			centroidOldAfter.ZRef()=oldCell->zCM;
			centMassOldAfter.XRef()=centroidOldAfter.X()/(float)(oldCell->volume);
			centMassOldAfter.YRef()=centroidOldAfter.Y()/(float)(oldCell->volume);
			centMassOldAfter.ZRef()=centroidOldAfter.Z()/(float)(oldCell->volume);


		}

	}

	if(newCell){

		centMassNewAfter.XRef()=newCell->xCM/(float)newCell->volume;
		centMassNewAfter.YRef()=newCell->yCM/(float)newCell->volume;
		centMassNewAfter.ZRef()=newCell->zCM/(float)newCell->volume;

		centroidNewAfter=precalculateCentroid(pt, newCell, 1,fieldDim,boundaryStrategy);
		centMassNewAfter.XRef()=centroidNewAfter.X()/(float)(newCell->volume+1);
		centMassNewAfter.YRef()=centroidNewAfter.Y()/(float)(newCell->volume+1);
		centMassNewAfter.ZRef()=centroidNewAfter.Z()/(float)(newCell->volume+1);

	}

	//will loop over neighbors of the oldCell and calculate Plasticity energy
	set<CurvatureTrackerData> * curvatureNeighborsTmpPtr;
	set<CurvatureTrackerData>::iterator sitr;

	float deltaL;
	float lAfter;
	float oldVol;
	float newVol;
	float nCellVol;

	CellG *rightNeighborOfOldCell=0;
	CellG *leftNeighborOfOldCell=0;

	CellG *rightRightNeighborOfOldCell=0;
	CellG *leftLeftNeighborOfOldCell=0;

	//cerr<<"energy before old cell section="<<energy<<endl;
	if(oldCell){
		//cerr<<"energy for old cell section"<<endl;
		oldVol=oldCell->volume;

		Vector3 rightCM;
		Vector3 rightRightCM;
		Vector3 rightRightRightCM;

		Vector3 rightCMAfter;
		Vector3 rightRightCMAfter;
		Vector3 rightRightRightCMAfter;

		Vector3 midCM;
		Vector3 midCMAfter;

		Vector3 leftCMAfter;
		Vector3 leftLeftCMAfter;
		Vector3 leftLeftLeftCMAfter;

		Vector3 leftCM;
		Vector3 leftLeftCM;
		Vector3 leftLeftLeftCM;




		const CellG * midCell=oldCell;
		CellG * leftCell=0;
		CellG * leftLeftCell=0;
		CellG * leftLeftLeftCell=0;

		CellG * rightCell=0;
		CellG * rightRightCell=0;
		CellG * rightRightRightCell=0;


		//pick neighbors of the old cell
		int count=0;
		curvatureNeighborsTmpPtr=&curvatureTrackerAccessor.get(oldCell->extraAttribPtr)->internalCurvatureNeighbors ;
		for (sitr=curvatureNeighborsTmpPtr->begin() ; sitr != curvatureNeighborsTmpPtr->end() ;++sitr){
			if (!count)
				rightCell=sitr->neighborAddress;
			else
				leftCell=sitr->neighborAddress;            
			++count;
		}

		//pick neighbors of the rightCell (of the oldCell)
		if (rightCell){
			curvatureNeighborsTmpPtr= &curvatureTrackerAccessor.get(rightCell->extraAttribPtr)->internalCurvatureNeighbors;
			int count=0;
			for (sitr=curvatureNeighborsTmpPtr->begin() ; sitr != curvatureNeighborsTmpPtr->end() ;++sitr){
				if (count<=1 && sitr->neighborAddress!=oldCell){
					rightRightCell=sitr->neighborAddress;
				}
				else if (count>1)
					break;//considering only 2 neighbors
				++count;
			}
		}

		//pick neighbors of the rightRightCell (of the oldCell)
		if (rightRightCell){
			curvatureNeighborsTmpPtr= &curvatureTrackerAccessor.get(rightRightCell->extraAttribPtr)->internalCurvatureNeighbors;
			int count=0;
			for (sitr=curvatureNeighborsTmpPtr->begin() ; sitr != curvatureNeighborsTmpPtr->end() ;++sitr){
				if (count<=1 && sitr->neighborAddress!=rightCell){
					rightRightRightCell=sitr->neighborAddress;
				}
				else if (count>1)
					break;//considering only 2 neighbors
				++count;
			}
		}


		//pick neighbors of the leftCell (of the oldCell)
		if (leftCell){
			curvatureNeighborsTmpPtr= &curvatureTrackerAccessor.get(leftCell->extraAttribPtr)->internalCurvatureNeighbors;
			int count=0;
			for (sitr=curvatureNeighborsTmpPtr->begin() ; sitr != curvatureNeighborsTmpPtr->end() ;++sitr){
				if (count<=1 && sitr->neighborAddress!=oldCell){
					leftLeftCell=sitr->neighborAddress;
				}
				else if (count>1)
					break;//considering only 2 neighbors
				++count;
			}
		}

		//pick neighbors of the leftLeftCell (of the oldCell)
		if (leftLeftCell){
			curvatureNeighborsTmpPtr= &curvatureTrackerAccessor.get(leftLeftCell->extraAttribPtr)->internalCurvatureNeighbors;
			int count=0;
			for (sitr=curvatureNeighborsTmpPtr->begin() ; sitr != curvatureNeighborsTmpPtr->end() ;++sitr){
				if (count<=1 && sitr->neighborAddress!=leftCell){
					leftLeftLeftCell=sitr->neighborAddress;
				}
				else if (count>1)
					break;//considering only 2 neighbors
				++count;
			}
		}


		rightNeighborOfOldCell=rightCell;
		leftNeighborOfOldCell=leftCell;
		rightRightNeighborOfOldCell=rightRightCell;
		leftLeftNeighborOfOldCell=leftLeftCell;

		//at this point we have all the cells which will participate in energy calculations so we have to calculate before and after flip values
		midCMAfter.SetXYZ(centMassOldAfter.x,centMassOldAfter.y,centMassOldAfter.z);
		midCM.SetXYZ(oldCell->xCM/oldVol,oldCell->yCM/oldVol,oldCell->zCM/oldVol);

		if(rightCell){
			if(rightCell==newCell){
				rightCM.SetXYZ(newCell->xCM/(float)newCell->volume,newCell->yCM/(float)newCell->volume,newCell->zCM/(float)newCell->volume);
				rightCMAfter.SetXYZ(centMassNewAfter.x,centMassNewAfter.y,centMassNewAfter.z);            
			}else{
				rightCM.SetXYZ(rightCell->xCM/(float)rightCell->volume,rightCell->yCM/(float)rightCell->volume,rightCell->zCM/(float)rightCell->volume);
				rightCMAfter=Vector3(rightCM);
			}
		}

		if(leftCell){
			if(leftCell==newCell){
				leftCM.SetXYZ(newCell->xCM/(float)newCell->volume,newCell->yCM/(float)newCell->volume,newCell->zCM/(float)newCell->volume);
				leftCMAfter.SetXYZ(centMassNewAfter.x,centMassNewAfter.y,centMassNewAfter.z);            
			}else{
				leftCM.SetXYZ(leftCell->xCM/(float)leftCell->volume,leftCell->yCM/(float)leftCell->volume,leftCell->zCM/(float)leftCell->volume);
				leftCMAfter=Vector3(leftCM);
			}
		}

		//this cell remains unaltered but participates in the calculations of the energy
		if(leftLeftCell){
			leftLeftCM.SetXYZ(leftLeftCell->xCM/(float)leftLeftCell->volume,leftLeftCell->yCM/(float)leftLeftCell->volume,leftLeftCell->zCM/(float)leftLeftCell->volume);
			leftLeftCMAfter=leftLeftCM;
		}
		//this cell remains unaltered but participate inthe calculations of the energy
		if(leftLeftLeftCell){
			leftLeftLeftCM.SetXYZ(leftLeftLeftCell->xCM/(float)leftLeftLeftCell->volume,leftLeftLeftCell->yCM/(float)leftLeftLeftCell->volume,leftLeftLeftCell->zCM/(float)leftLeftLeftCell->volume);
			leftLeftLeftCMAfter=leftLeftLeftCM;
		}


		//this cell remains unaltered but participate inthe calculations of the energy
		if(rightRightCell){
			rightRightCM.SetXYZ(rightRightCell->xCM/(float)rightRightCell->volume,rightRightCell->yCM/(float)rightRightCell->volume,rightRightCell->zCM/(float)rightRightCell->volume);
			rightRightCMAfter=rightRightCM;
		}

		//this cell remains unaltered but participate inthe calculations of the energy
		if(rightRightRightCell){
			rightRightRightCM.SetXYZ(rightRightRightCell->xCM/(float)rightRightRightCell->volume,rightRightRightCell->yCM/(float)rightRightRightCell->volume,rightRightRightCell->zCM/(float)rightRightRightCell->volume);
			rightRightRightCMAfter=rightRightRightCM;
		}


		double lambda;

		//calculate change in curvature energy
		//cerr<<"pt="<<pt<<endl;
		if (midCell->volume>1){ 
			if (leftLeftLeftCell && leftLeftCell && leftCell){                

				lambda=internalCurvatureParamsArray[leftLeftLeftCell->type][leftLeftCell->type].lambdaCurvature;
				double energyPartial=lambda*(calculateInverseCurvatureSquare(leftLeftLeftCMAfter,leftLeftCMAfter,leftCMAfter)-calculateInverseCurvatureSquare(leftLeftLeftCM,leftLeftCM,leftCM));
				energy+=energyPartial;
				//if (newCell && oldCell){
				//cerr<<"lllCMAfter="<<leftLeftLeftCMAfter<<" lllBefore="<<leftLeftLeftCM<<endl;
				//cerr<<"llCMAfter="<<leftLeftCMAfter<<" llBefore="<<leftLeftCM<<endl;
				//cerr<<"lCMAfter="<<leftCMAfter<<" lBefore="<<leftCM<<endl;
				//
				//cerr<<"energy partial="<<energyPartial<<endl;
				//}
			}

			if (leftLeftCell && leftCell && midCell){


				lambda=internalCurvatureParamsArray[leftLeftCell->type][leftCell->type].lambdaCurvature;
				energy+=lambda*(calculateInverseCurvatureSquare(leftLeftCMAfter,leftCMAfter,midCMAfter)-calculateInverseCurvatureSquare(leftLeftCM,leftCM,midCM));


				//cerr<<"llCMAfter="<<leftLeftCMAfter<<" llBefore="<<leftLeftCM<<endl;
				//cerr<<"lCMAfter="<<leftCMAfter<<" lBefore="<<leftCM<<endl;
				//cerr<<"mCMAfter="<<midCMAfter<<" mBefore="<<midCM<<endl;
				//cerr<<"energy partial="<<energy<<endl;

			}

			if (leftCell && midCell && rightCell){

				lambda=internalCurvatureParamsArray[leftCell->type][midCell->type].lambdaCurvature;
				energy+=lambda*(calculateInverseCurvatureSquare(leftCMAfter,midCMAfter,rightCMAfter)-calculateInverseCurvatureSquare(leftCM,midCM,rightCM));


				//cerr<<"lCMAfter="<<leftCMAfter<<" lBefore="<<leftCM<<endl;
				//cerr<<"mCMAfter="<<midCMAfter<<" mBefore="<<midCM<<endl;
				//cerr<<"rCMAfter="<<rightCMAfter<<" rBefore="<<rightCM<<endl;
				//cerr<<"energy partial="<<energy<<endl;

			}

			if ( midCell && rightCell && rightRightCell){                

				lambda=internalCurvatureParamsArray[midCell->type][rightCell->type].lambdaCurvature;
				energy+=lambda*(calculateInverseCurvatureSquare(midCMAfter,rightCMAfter,rightRightCMAfter)-calculateInverseCurvatureSquare(midCM,rightCM,rightRightCM));


				//cerr<<"mCMAfter="<<midCMAfter<<" mBefore="<<midCM<<endl;
				//cerr<<"rCMAfter="<<rightCMAfter<<" rBefore="<<rightCM<<endl;
				//cerr<<"rrCMAfter="<<rightRightCMAfter<<" rrBefore="<<rightRightCM<<endl;
				//cerr<<"energy partial="<<energy<<endl;

			}
			if (  rightCell && rightRightCell && rightRightRightCell){                

				lambda=internalCurvatureParamsArray[rightCell->type][rightRightCell->type].lambdaCurvature;
				double energyPartial=lambda*(calculateInverseCurvatureSquare(rightCMAfter,rightRightCMAfter,rightRightRightCMAfter)-calculateInverseCurvatureSquare(rightCM,rightRightCM,rightRightRightCM));
				energy+=energyPartial;
				//if (newCell && oldCell){
				//cerr<<"rCMAfter="<<rightCMAfter<<" rBefore="<<rightCM<<endl;
				//cerr<<"rrCMAfter="<<rightRightCMAfter<<" rrBefore="<<rightRightCM<<endl;
				//cerr<<"rrrCMAfter="<<rightRightRightCMAfter<<" rtrBefore="<<rightRightRightCM<<endl;
				//cerr<<"energy partial="<<energyPartial<<endl;
				//}

			}


		}else{ // midCell=oldCell is about to disappear

			if (leftLeftLeftCell && leftLeftCell && leftCell){                

				lambda=internalCurvatureParamsArray[leftLeftLeftCell->type][leftLeftCell->type].lambdaCurvature;
				energy+=lambda*(calculateInverseCurvatureSquare(leftLeftLeftCMAfter,leftLeftCMAfter,leftCMAfter)-calculateInverseCurvatureSquare(leftLeftLeftCM,leftLeftCM,leftCM));
			}

			if (leftLeftCell && leftCell && midCell){

				lambda=internalCurvatureParamsArray[leftLeftCell->type][leftCell->type].lambdaCurvature;
				energy-=lambda*calculateInverseCurvatureSquare(leftLeftCM,leftCM,midCM);
				if(rightCell)
					energy+=calculateInverseCurvatureSquare(leftLeftCMAfter,leftCMAfter,rightCMAfter);
			}

			if (leftCell && midCell && rightCell){

				lambda=internalCurvatureParamsArray[leftCell->type][midCell->type].lambdaCurvature;
				energy-=lambda*calculateInverseCurvatureSquare(leftCM,midCM,rightCM);
				if(rightRightCell)
					energy+=calculateInverseCurvatureSquare(leftCMAfter,rightCMAfter,rightRightCMAfter);
			}else if (midCell && rightCell && rightRightCell){
				lambda=internalCurvatureParamsArray[midCell->type][rightCell->type].lambdaCurvature;
				energy-=lambda*calculateInverseCurvatureSquare(midCM,rightCM,rightRightCM);
				if(leftCell)
					energy+=calculateInverseCurvatureSquare(leftCMAfter,rightCMAfter,rightRightCMAfter);				
			}
			if (  rightCell && rightRightCell && rightRightRightCell){                

				lambda=internalCurvatureParamsArray[midCell->type][rightCell->type].lambdaCurvature;
				energy+=lambda*(calculateInverseCurvatureSquare(midCMAfter,rightCMAfter,rightRightCMAfter)-calculateInverseCurvatureSquare(midCM,rightCM,rightRightCM));
			}


		}

	}

	if(newCell && !oldCell){
		//cerr<<"energy for old cell section"<<endl;
		newVol=newCell->volume;
		Vector3 rightCM;
		Vector3 rightRightCM;
		Vector3 rightRightRightCM;

		Vector3 rightCMAfter;
		Vector3 rightRightCMAfter;
		Vector3 rightRightRightCMAfter;

		Vector3 midCM;
		Vector3 midCMAfter;

		Vector3 leftCMAfter;
		Vector3 leftLeftCMAfter;
		Vector3 leftLeftLeftCMAfter;

		Vector3 leftCM;
		Vector3 leftLeftCM;
		Vector3 leftLeftLeftCM;




		const CellG * midCell=newCell;
		CellG * leftCell=0;
		CellG * leftLeftCell=0;
		CellG * leftLeftLeftCell=0;

		CellG * rightCell=0;
		CellG * rightRightCell=0;
		CellG * rightRightRightCell=0;

		//pick neighbors of the new cell
		int count=0;
		curvatureNeighborsTmpPtr=&curvatureTrackerAccessor.get(newCell->extraAttribPtr)->internalCurvatureNeighbors ;
		for (sitr=curvatureNeighborsTmpPtr->begin() ; sitr != curvatureNeighborsTmpPtr->end() ;++sitr){
			if (!count)
				rightCell=sitr->neighborAddress;
			else
				leftCell=sitr->neighborAddress;            
			++count;
		}

		//pick neighbors of the rightCell (of the newCell)
		if (rightCell){
			curvatureNeighborsTmpPtr= &curvatureTrackerAccessor.get(rightCell->extraAttribPtr)->internalCurvatureNeighbors;
			int count=0;
			for (sitr=curvatureNeighborsTmpPtr->begin() ; sitr != curvatureNeighborsTmpPtr->end() ;++sitr){
				if (count<=1 && sitr->neighborAddress!=newCell){
					rightRightCell=sitr->neighborAddress;
				}
				else if (count>1)
					break;//considering only 2 neighbors
				++count;
			}
		}

		//   //pick neighbors of the rightRightCell (of the newCell)
		//   if (rightRightCell){
		//       curvatureNeighborsTmpPtr= &curvatureTrackerAccessor.get(rightRightCell->extraAttribPtr)->internalCurvatureNeighbors;
		//       int count=0;
		//       for (sitr=curvatureNeighborsTmpPtr->begin() ; sitr != curvatureNeighborsTmpPtr->end() ;++sitr){
		//         if (count<=1 && sitr->neighborAddress!=rightCell){
		//             rightRightRightCell=sitr->neighborAddress;
		//}
		//         else if (count>1)
		//             break;//considering only 2 neighbors
		//         ++count;
		//       }
		//   }

		//pick neighbors of the leftCell (of the newCell)
		if (leftCell){
			curvatureNeighborsTmpPtr= &curvatureTrackerAccessor.get(leftCell->extraAttribPtr)->internalCurvatureNeighbors;
			int count=0;
			for (sitr=curvatureNeighborsTmpPtr->begin() ; sitr != curvatureNeighborsTmpPtr->end() ;++sitr){
				if (count<=1 && sitr->neighborAddress!=newCell){
					leftLeftCell=sitr->neighborAddress;
				}
				else if (count>1)
					break;//considering only 2 neighbors
				++count;
			}
		}

		//   //pick neighbors of the leftLeftCell (of the newCell)
		//   if (leftLeftCell){
		//       curvatureNeighborsTmpPtr= &curvatureTrackerAccessor.get(leftLeftCell->extraAttribPtr)->internalCurvatureNeighbors;
		//       int count=0;
		//       for (sitr=curvatureNeighborsTmpPtr->begin() ; sitr != curvatureNeighborsTmpPtr->end() ;++sitr){
		//         if (count<=1 && sitr->neighborAddress!=leftCell){
		//             leftLeftLeftCell=sitr->neighborAddress;
		//}
		//         else if (count>1)
		//             break;//considering only 2 neighbors
		//         ++count;
		//       }
		//   }

		//at this point we have all the cells which will participate in energy calculations so we have to calculate before and after flip values
		midCMAfter.SetXYZ(centMassNewAfter.x,centMassNewAfter.y,centMassNewAfter.z);
		midCM.SetXYZ(newCell->xCM/newVol,newCell->yCM/newVol,newCell->zCM/newVol);
		if(rightCell){
			rightCM.SetXYZ(rightCell->xCM/(float)rightCell->volume,rightCell->yCM/(float)rightCell->volume,rightCell->zCM/(float)rightCell->volume);
			rightCMAfter=Vector3(rightCM);          
		}

		if(leftCell){
			leftCM.SetXYZ(leftCell->xCM/(float)leftCell->volume,leftCell->yCM/(float)leftCell->volume,leftCell->zCM/(float)leftCell->volume);
			leftCMAfter=Vector3(leftCM);
		}




		//this cell remains unaltered but participate inthe calculations of the energy
		if(leftLeftCell){
			leftLeftCM.SetXYZ(leftLeftCell->xCM/(float)leftLeftCell->volume,leftLeftCell->yCM/(float)leftLeftCell->volume,leftLeftCell->zCM/(float)leftLeftCell->volume);
			leftLeftCMAfter=leftLeftCM;
		}

		//this cell remains unaltered but participate inthe calculations of the energy
		if(rightRightCell){
			rightRightCM.SetXYZ(rightRightCell->xCM/(float)rightRightCell->volume,rightRightCell->yCM/(float)rightRightCell->volume,rightRightCell->zCM/(float)rightRightCell->volume);
			rightRightCMAfter=rightRightCM;
		}


		double lambda;

		//calculate change in curvature energy
		if (leftLeftCell && leftCell && midCell){
			// lambda=curvatureTrackerAccessor.get(leftLeftCell->extraAttribPtr)->internalCurvatureNeighbors.begin()->lambdaCurvature;
			lambda=internalCurvatureParamsArray[leftLeftCell->type][leftCell->type].lambdaCurvature;
			energy+=lambda*(calculateInverseCurvatureSquare(leftLeftCMAfter,leftCMAfter,midCMAfter)-calculateInverseCurvatureSquare(leftLeftCM,leftCM,midCM));
		}

		if (leftCell && midCell && rightCell){
			// lambda=curvatureTrackerAccessor.get(leftCell->extraAttribPtr)->internalCurvatureNeighbors.begin()->lambdaCurvature;
			lambda=internalCurvatureParamsArray[leftCell->type][midCell->type].lambdaCurvature;
			energy+=lambda*(calculateInverseCurvatureSquare(leftCMAfter,midCMAfter,rightCMAfter)-calculateInverseCurvatureSquare(leftCM,midCM,rightCM));
		}

		if ( midCell && rightCell && rightRightCell){
			// lambda=curvatureTrackerAccessor.get(midCell->extraAttribPtr)->internalCurvatureNeighbors.begin()->lambdaCurvature;
			lambda=internalCurvatureParamsArray[midCell->type][rightCell->type].lambdaCurvature;
			energy+=lambda*(calculateInverseCurvatureSquare(midCMAfter,rightCMAfter,rightRightCMAfter)-calculateInverseCurvatureSquare(midCM,rightCM,rightRightCM));
		}



	}
	//cerr<<"DELTA E="<<energy<<endl;
	return energy;


	// if(newCell && oldCell){ //in this case we have to calculate change of energy for one triple of center of masses only 
	//                          //first possibility (rightRightRight , rightRight, right)
	//                         //second possibility (leftLeftLeft , leftLeft, left)

	//    CellG * rightCell=0;
	//    const CellG * midCell=newCell;
	//    CellG * leftCell=0;
	//    CellG * leftLeftCell=0;
	//    CellG * rightRightCell=0;
	//    
	//    //pick neighbors of the new cell
	//    int count=0;
	// curvatureNeighborsTmpPtr= &curvatureTrackerAccessor.get(newCell->extraAttribPtr)->internalCurvatureNeighbors;
	//    for (sitr=curvatureNeighborsTmpPtr->begin() ; sitr != curvatureNeighborsTmpPtr->end() ;++sitr){
	//      if (!count)
	//          rightCell=sitr->neighborAddress;
	//      else
	//          leftCell=sitr->neighborAddress;            
	//      ++count;
	//    }
	//    
	//    //pick neighbors of the rightCell (of the newCell)
	//    if (rightCell){
	//        curvatureNeighborsTmpPtr= &curvatureTrackerAccessor.get(rightCell->extraAttribPtr)->internalCurvatureNeighbors;
	//        int count=0;
	//        for (sitr=curvatureNeighborsTmpPtr->begin() ; sitr != curvatureNeighborsTmpPtr->end() ;++sitr){
	//          if (count<=1 && sitr->neighborAddress!=newCell){
	//              rightRightCell=sitr->neighborAddress;
	//	}
	//          else if (count>1)
	//              break;//considering only 2 neighbors
	//          ++count;
	//        }
	//    }

	//    //pick neighbors of the leftCell (of the newCell)
	//    if (leftCell){
	//        curvatureNeighborsTmpPtr= &curvatureTrackerAccessor.get(leftCell->extraAttribPtr)->internalCurvatureNeighbors;
	//        int count=0;
	//        for (sitr=curvatureNeighborsTmpPtr->begin() ; sitr != curvatureNeighborsTmpPtr->end() ;++sitr){
	//          if (count<=1 && sitr->neighborAddress!=newCell){
	//              leftLeftCell=sitr->neighborAddress;
	//	}
	//          else if (count>1)
	//              break;//considering only 2 neighbors
	//          ++count;
	//        }
	//    }
	//Coordinates3D<float> centMassNewBefore;
	//centMassNewBefore.x=newCell->xCM/(float)newCell->volume;
	//centMassNewBefore.y=newCell->yCM/(float)newCell->volume;
	//centMassNewBefore.z=newCell->zCM/(float)newCell->volume;

	//      double lambda;
	//cerr<<"rightNeighborOfOldCell.volume="<<rightNeighborOfOldCell->volume<<" newCell->volume="<<newCell->volume<<endl;		
	//      if (rightNeighborOfOldCell==newCell && rightRightNeighborOfOldCell){
	//          //pick an neighbor of rightRightNeighborOfOldCell which is different than newCell
	//          CellG *rightRightRightCell=0;
	//            curvatureNeighborsTmpPtr= &curvatureTrackerAccessor.get(rightRightNeighborOfOldCell->extraAttribPtr)->internalCurvatureNeighbors;
	//            int count=0;
	//            for (sitr=curvatureNeighborsTmpPtr->begin() ; sitr != curvatureNeighborsTmpPtr->end() ;++sitr){
	//              if (count<=1 && sitr->neighborAddress!=newCell){
	//                  rightRightRightCell=sitr->neighborAddress;
	//		}
	//              else if (count>1)
	//                  break;//considering only 2 neighbors
	//              ++count;
	//            }

	//           cerr<<"BEFORE if rightRightRightCell="<<rightRightRightCell<<endl;

	//           if(rightRightRightCell){
	//		cerr<<"rightRightRightCell->volume="<<rightRightRightCell->volume<<endl;
	//              Vector3 rightRightRightCMBefore(rightRightRightCell->xCM/float(rightRightRightCell->volume),rightRightRightCell->yCM/float(rightRightRightCell->volume),rightRightRightCell->zCM/float(rightRightRightCell->volume));                
	//              Vector3 rightRightRightCMAfter(rightRightRightCMBefore); 
	//		cerr<<"rightRightCell="<<rightRightCell<<endl;
	//	    cerr<<" vol="<<rightRightCell->volume<<endl;

	//              Vector3 rightRightCMBefore(rightRightCell->xCM/float(rightRightCell->volume),rightRightCell->yCM/float(rightRightCell->volume),rightRightCell->zCM/float(rightRightCell->volume));
	//              Vector3 rightRightCMAfter(rightRightCMBefore);
	//              Vector3 rightCMBefore(centMassNewBefore.x,centMassNewBefore.y,centMassNewBefore.z);
	//              Vector3 rightCMAfter(centMassNewAfter.x,centMassNewAfter.y,centMassNewAfter.z) ;               
	//              
	//              // lambda=curvatureTrackerAccessor.get(rightRightRightCell->extraAttribPtr)->internalCurvatureNeighbors.begin()->lambdaCurvature;
	//              lambda=internalCurvatureParamsArray[rightRightRightCell->type][rightRightCell->type].lambdaCurvature;
	//		cerr<<"lambda="<<lambda<<endl;
	//              energy+=lambda*(calculateInverseCurvatureSquare(rightRightRightCMAfter,rightRightCMAfter,rightCMAfter)-calculateInverseCurvatureSquare(rightRightRightCMBefore,rightRightCMBefore,rightCMBefore));
	//           }
	//}  
	//  //    else if (leftNeighborOfOldCell==newCell && leftLeftNeighborOfOldCell){
	//  //        //pick an neighbor of leftLeftNeighborOfOldCell which is different than newCell
	//  //          CellG *leftLeftLeftCell=0;
	//  //          curvatureNeighborsTmpPtr= &curvatureTrackerAccessor.get(leftLeftNeighborOfOldCell->extraAttribPtr)->internalCurvatureNeighbors;
	//  //          int count=0;
	//	 // leftLeftLeftCell=0;
	//  //          for (sitr=curvatureNeighborsTmpPtr->begin() ; sitr != curvatureNeighborsTmpPtr->end() ;++sitr){
	//  //            if (count<=1 && sitr->neighborAddress!=newCell){
	//  //                leftLeftLeftCell=sitr->neighborAddress;
	//		//}
	//  //            else if (count>1)
	//  //                break;//considering only 2 neighbors
	//  //            ++count;
	//  //          }
	//  //          
	//  //         if(leftLeftLeftCell){
	//  //            Vector3 leftLeftLeftCMBefore(leftLeftLeftCell->xCM/float(leftLeftLeftCell->volume),leftLeftLeftCell->yCM/float(leftLeftLeftCell->volume),leftLeftLeftCell->zCM/float(leftLeftLeftCell->volume));                
	//  //            Vector3 leftLeftLeftCMAfter(leftLeftLeftCMBefore);
	//  //            
	//  //            Vector3 leftLeftCMBefore(leftLeftCell->xCM/float(leftLeftCell->volume),leftLeftCell->yCM/float(leftLeftCell->volume),leftLeftCell->zCM/float(leftLeftCell->volume));
	//  //            Vector3 leftLeftCMAfter(leftLeftCMBefore);
	//  //            Vector3 leftCMBefore(centMassNewBefore.x,centMassNewBefore.y,centMassNewBefore.z);
	//  //            Vector3 leftCMAfter(centMassNewAfter.x,centMassNewAfter.y,centMassNewAfter.z)  ;              
	//  //            
	//  //            // lambda=curvatureTrackerAccessor.get(leftLeftLeftCell->extraAttribPtr)->internalCurvatureNeighbors.begin()->lambdaCurvature;
	//  //            lambda=internalCurvatureParamsArray[leftLeftLeftCell->type][leftLeftCell->type].lambdaCurvature;
	//  //            energy+=lambda*(calculateInverseCurvatureSquare(leftLeftLeftCMAfter,leftLeftCMAfter,leftCMAfter)-calculateInverseCurvatureSquare(leftLeftLeftCMBefore,leftLeftCMBefore,leftCMBefore));
	//  //         }
	//  //    
	//  //    }
	//    
	// }


}

void CurvaturePlugin::field3DChange(const Point3D &pt, CellG *newCell,CellG *oldCell){

	int currentWorkNodeNumber=pUtils->getCurrentWorkNodeNumber();	    
	short &  newJunctionInitiatedFlagWithinCluster = newJunctionInitiatedFlagWithinClusterVec[currentWorkNodeNumber];
	CellG * & newNeighbor=newNeighborVec[currentWorkNodeNumber];

	if (newJunctionInitiatedFlagWithinCluster){
		double xCMNew=newCell->xCM/float(newCell->volume);
		double yCMNew=newCell->yCM/float(newCell->volume);
		double zCMNew=newCell->zCM/float(newCell->volume);

		double xCMNeighbor=newNeighbor->xCM/float(newNeighbor->volume);
		double yCMNeighbor=newNeighbor->yCM/float(newNeighbor->volume);
		double zCMNeighbor=newNeighbor->zCM/float(newNeighbor->volume);

		double distance=distInvariantCM(xCMNew,yCMNew,zCMNew,xCMNeighbor,yCMNeighbor,zCMNeighbor,fieldDim,boundaryStrategy);
		//double distance=dist(xCMNew,yCMNew,zCMNew,xCMNeighbor,yCMNeighbor,zCMNeighbor);

		//if (curvatureTypes.size()==0||(curvatureTypes.find(newNeighbor->type)!=curvatureTypes.end() && curvatureTypes.find(newCell->type)!=curvatureTypes.end())){
		if (functionType==BYCELLTYPE){
			//cerr<<"adding internal junction between "<<newCell<<" and "<<newNeighbor<<endl;
			CurvatureTrackerData ctd=internalCurvatureParamsArray[newCell->type][newNeighbor->type];

			ctd.neighborAddress=newNeighbor;

			curvatureTrackerAccessor.get(newCell->extraAttribPtr)->internalCurvatureNeighbors.		
				insert(CurvatureTrackerData(ctd));

			//cerr<<"newCell="<<newCell<< " FTTPDs="<<CurvatureTrackerAccessor.get(newCell->extraAttribPtr)->internalCurvatureNeighbors.size()<<endl;

			ctd.neighborAddress=newCell;
			curvatureTrackerAccessor.get(newNeighbor->extraAttribPtr)->internalCurvatureNeighbors.
				insert(CurvatureTrackerData(ctd));

			//cerr<<"newNeighbor="<<newNeighbor<<" FTTPDs="<<CurvatureTrackerAccessor.get(newNeighbor->extraAttribPtr)->internalCurvatureNeighbors.size()<<endl;

		}

		//}

		return;
	}


	// oldCell is about to disappear so we need to remove all references to it from curvature neighbors
	if(oldCell && oldCell->volume==0){

		//cerr<<"oldCell.id="<<oldCell->id<<" is to be removed"<<endl;
		set<CurvatureTrackerData>::iterator sitr;		
		//go over compartments
		set<CurvatureTrackerData> & internalCurvatureNeighbors=curvatureTrackerAccessor.get(oldCell->extraAttribPtr)->internalCurvatureNeighbors;

		vector<pair<CellG*, CurvatureTrackerData> > oldCellNeighborVec;

		for(sitr=internalCurvatureNeighbors.begin() ; sitr != internalCurvatureNeighbors.end() ; ++sitr){
			//cerr<<" REMOVING NEIGHBOR "<<oldCell->id<<" from list of "<<sitr->neighborAddress->id<<endl;
			std::set<CurvatureTrackerData> & curvatureNeighborsRemovedNeighbor=
				curvatureTrackerAccessor.get(sitr->neighborAddress->extraAttribPtr)->internalCurvatureNeighbors;

			CurvatureTrackerData oldCellNeighbor(*sitr);

			oldCellNeighborVec.push_back(make_pair(sitr->neighborAddress,oldCellNeighbor));
			curvatureNeighborsRemovedNeighbor.erase(CurvatureTrackerData(oldCell));			            
		}

		oldCellNeighborVec[0].second.neighborAddress=oldCellNeighborVec[1].first;        
		//std::set<CurvatureTrackerData> & curvatureNeighbor_0=


		curvatureTrackerAccessor.get(oldCellNeighborVec[0].first->extraAttribPtr)->internalCurvatureNeighbors.insert(oldCellNeighborVec[0].second);

		oldCellNeighborVec[1].second.neighborAddress=oldCellNeighborVec[0].first;

		/*std::set<CurvatureTrackerData> & curvatureNeighbor_1=*/
		curvatureTrackerAccessor.get(oldCellNeighborVec[1].first->extraAttribPtr)->internalCurvatureNeighbors.insert(oldCellNeighborVec[1].second);


	}
}


//void CurvaturePlugin::setPlasticityParameters(CellG * _cell1,CellG * _cell2,double _lambda, double _targetDistance){
//	
//	std::set<CurvatureTrackerData> & plastNeighbors1=curvatureTrackerAccessor.get(_cell1->extraAttribPtr)->CurvatureNeighbors;	
//	std::set<CurvatureTrackerData>::iterator sitr1;
//	sitr1=plastNeighbors1.find(CurvatureTrackerData(_cell2));
//	if(sitr1!=plastNeighbors1.end()){
//                //dirty solution to manipulate class stored in a set
//		(const_cast<CurvatureTrackerData & >(*sitr1)).lambdaDistance=_lambda;
//		if(_targetDistance!=0.0){
//			(const_cast<CurvatureTrackerData & >(*sitr1)).targetDistance=_targetDistance;
//		}
//		//have to change entries in _cell2 for curvature data associated with _cell1
//		std::set<CurvatureTrackerData> & plastNeighbors2=CurvatureTrackerAccessor.get(_cell2->extraAttribPtr)->CurvatureNeighbors;	
//		std::set<CurvatureTrackerData>::iterator sitr2;
//		sitr2=plastNeighbors1.find(CurvatureTrackerData(_cell1));
//		if(sitr2!=plastNeighbors2.end()){
//			(const_cast<CurvatureTrackerData & >(*sitr2)).lambdaDistance=_lambda;
//			if(_targetDistance!=0.0){
//				(const_cast<CurvatureTrackerData & >(*sitr2)).targetDistance=_targetDistance;
//		}
//			
//		}
//
//	}
//}

//double CurvaturePlugin::getPlasticityParametersLambdaDistance(CellG * _cell1,CellG * _cell2){
//
//	std::set<CurvatureTrackerData>::iterator sitr1=CurvatureTrackerAccessor.get(_cell1->extraAttribPtr)->CurvatureNeighbors.find(CurvatureTrackerData(_cell2));
//	if(sitr1!=CurvatureTrackerAccessor.get(_cell1->extraAttribPtr)->CurvatureNeighbors.end()){
//		return sitr1->lambdaDistance;
//	}else{
//		return 0.0;
//	}
//}

//double CurvaturePlugin::getPlasticityParametersTargetDistance(CellG * _cell1,CellG * _cell2){
//
//	std::set<CurvatureTrackerData>::iterator sitr1=CurvatureTrackerAccessor.get(_cell1->extraAttribPtr)->CurvatureNeighbors.find(CurvatureTrackerData(_cell2));
//	if(sitr1!=CurvatureTrackerAccessor.get(_cell1->extraAttribPtr)->CurvatureNeighbors.end()){
//		return sitr1->targetDistance;
//	}else{
//		return 0.0;
//	}
//}



std::string CurvaturePlugin::steerableName(){return "Curvature";}
std::string CurvaturePlugin::toString(){return steerableName();}


int CurvaturePlugin::getIndex(const int type1, const int type2) const {
	if (type1 < type2) return ((type1 + 1) | ((type2 + 1) << 16));
	else return ((type2 + 1) | ((type1 + 1) << 16));
}