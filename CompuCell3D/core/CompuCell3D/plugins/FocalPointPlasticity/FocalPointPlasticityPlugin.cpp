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

#include "FocalPointPlasticityPlugin.h"


FocalPointPlasticityPlugin::FocalPointPlasticityPlugin():pUtils(0),xmlData(0)   {
	lambda=0.0;
	activationEnergy=0.0;

	targetDistance=0.0;


	maxDistance=1000.0;

	diffEnergyFcnPtr=&FocalPointPlasticityPlugin::diffEnergyByType;

	//setting default elastic link constituent law
	constituentLawFcnPtr=&FocalPointPlasticityPlugin::elasticLinkConstituentLaw;

	functionType=BYCELLTYPE;
	neighborOrder=1;
}

FocalPointPlasticityPlugin::~FocalPointPlasticityPlugin() {

}

void FocalPointPlasticityPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
	sim = simulator;
	potts=simulator->getPotts();
	fieldDim=potts->getCellFieldG()->getDim();
	xmlData=_xmlData;
	simulator->getPotts()->registerEnergyFunctionWithName(this,toString());
	simulator->registerSteerableObject(this);

	///will register FocalPointBoundaryPixelTracker here
	//BasicClassAccessorBase * cellFocalPointBoundaryPixelTrackerAccessorPtr=&focalPointPlasticityTrackerAccessor;
	///************************************************************************************************  
	///REMARK. HAVE TO USE THE SAME BASIC CLASS ACCESSOR INSTANCE THAT WAS USED TO REGISTER WITH FACTORY
	///************************************************************************************************  

	bool pluginAlreadyRegisteredFlag;
	Plugin *plugin=Simulator::pluginManager.get("CenterOfMass",&pluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded
	if(!pluginAlreadyRegisteredFlag)
		plugin->init(simulator);

	//first need to register center of mass plugin and then register FocalPointPlasticity
	potts->getCellFactoryGroupPtr()->registerClass(&focalPointPlasticityTrackerAccessor);
	potts->registerCellGChangeWatcher(this);  

	pUtils=simulator->getParallelUtils();
	unsigned int maxNumberOfWorkNodes=pUtils->getMaxNumberOfWorkNodesPotts();        
	newJunctionInitiatedFlagVec.assign(maxNumberOfWorkNodes,false);
	newJunctionInitiatedFlagWithinClusterVec.assign(maxNumberOfWorkNodes,false);
	newNeighborVec.assign(maxNumberOfWorkNodes,0);

}

void FocalPointPlasticityPlugin::extraInit(Simulator *simulator){
	update(xmlData,true);
}

void FocalPointPlasticityPlugin::handleEvent(CC3DEvent & _event){
    if (_event.id==CHANGE_NUMBER_OF_WORK_NODES){    
    	unsigned int maxNumberOfWorkNodes=pUtils->getMaxNumberOfWorkNodesPotts();        
        newJunctionInitiatedFlagVec.assign(maxNumberOfWorkNodes,false);
        newJunctionInitiatedFlagWithinClusterVec.assign(maxNumberOfWorkNodes,false);
        newNeighborVec.assign(maxNumberOfWorkNodes,0);

    	update(xmlData);
    }

}


void FocalPointPlasticityPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

	automaton = potts->getAutomaton();

	set<unsigned char> cellTypesSet;
	set<unsigned char> internalCellTypesSet;
	set<unsigned char> typeSpecCellTypesSet;
	set<unsigned char> internalTypeSpecCellTypesSet;

	plastParams.clear();
	internalPlastParams.clear();
	typeSpecificPlastParams.clear();
	internalTypeSpecificPlastParams.clear();

	ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)

		CC3DXMLElementList plastParamVec=_xmlData->getElements("Parameters");
	if (plastParamVec.size()>0){
		functionType=BYCELLTYPE;
	}

	if(_xmlData->getFirstElement("Local")){
		diffEnergyFcnPtr=&FocalPointPlasticityPlugin::diffEnergyLocal;
		functionType=BYCELLID;
		//return;
	}

	if(_xmlData->getFirstElement("NeighborOrder")){
		neighborOrder=_xmlData->getFirstElement("NeighborOrder")->getInt();
	}

	for (int i = 0 ; i<plastParamVec.size(); ++i){

		FocalPointPlasticityTrackerData fpptd;

		char type1 = automaton->getTypeId(plastParamVec[i]->getAttribute("Type1"));
		char type2 = automaton->getTypeId(plastParamVec[i]->getAttribute("Type2"));

		int index = getIndex(type1, type2);

		plastParams_t::iterator it = plastParams.find(index);
		ASSERT_OR_THROW(string("Plasticity parameters for ") + type1 + " " + type2 +
			" already set!", it == plastParams.end());

		if(plastParamVec[i]->getFirstElement("Lambda"))
			fpptd.lambdaDistance=plastParamVec[i]->getFirstElement("Lambda")->getDouble();

		if(plastParamVec[i]->getFirstElement("TargetDistance"))
			fpptd.targetDistance=plastParamVec[i]->getFirstElement("TargetDistance")->getDouble();

		if(plastParamVec[i]->getFirstElement("ActivationEnergy")){
			fpptd.activationEnergy=plastParamVec[i]->getFirstElement("ActivationEnergy")->getDouble();
		}

		if(plastParamVec[i]->getFirstElement("MaxDistance"))
			fpptd.maxDistance=plastParamVec[i]->getFirstElement("MaxDistance")->getDouble();

		if(plastParamVec[i]->getFirstElement("MaxNumberOfJunctions")){
			CC3DXMLElement * maxNumberOfJunctionsElement=plastParamVec[i]->getFirstElement("MaxNumberOfJunctions");
			fpptd.maxNumberOfJunctions=maxNumberOfJunctionsElement->getInt();

			//if(maxNumberOfJunctionsElement->findAttribute("NeighborOrder"))				
			//	fpptd.neighborOrder=maxNumberOfJunctionsElement->getAttributeAsUInt("NeighborOrder");
		}

		plastParams[index] = fpptd;

		//inserting all the types to the set (duplicate are automatically eleminated) to figure out max value of type Id
		cellTypesSet.insert(type1);
		cellTypesSet.insert(type2);			
	}


	// extracting internal parameters - used with compartmental cells
	CC3DXMLElementList internalPlastParamVec=_xmlData->getElements("InternalParameters");
	for (int i = 0 ; i<internalPlastParamVec.size(); ++i){

		FocalPointPlasticityTrackerData fpptd;

		char type1 = automaton->getTypeId(internalPlastParamVec[i]->getAttribute("Type1"));
		char type2 = automaton->getTypeId(internalPlastParamVec[i]->getAttribute("Type2"));

		int index = getIndex(type1, type2);

		plastParams_t::iterator it = internalPlastParams.find(index);
		ASSERT_OR_THROW(string("Internal plasticity parameters for ") + type1 + " " + type2 +
			" already set!", it == internalPlastParams.end());


		if(internalPlastParamVec[i]->getFirstElement("Lambda"))
			fpptd.lambdaDistance=internalPlastParamVec[i]->getFirstElement("Lambda")->getDouble();

		if(internalPlastParamVec[i]->getFirstElement("TargetDistance"))
			fpptd.targetDistance=internalPlastParamVec[i]->getFirstElement("TargetDistance")->getDouble();

		if(internalPlastParamVec[i]->getFirstElement("ActivationEnergy"))
			fpptd.activationEnergy=internalPlastParamVec[i]->getFirstElement("ActivationEnergy")->getDouble();

		if(internalPlastParamVec[i]->getFirstElement("MaxDistance"))
			fpptd.maxDistance=internalPlastParamVec[i]->getFirstElement("MaxDistance")->getDouble();

		if(internalPlastParamVec[i]->getFirstElement("MaxNumberOfJunctions")){
			CC3DXMLElement * maxNumberOfJunctionsElement=internalPlastParamVec[i]->getFirstElement("MaxNumberOfJunctions");
			fpptd.maxNumberOfJunctions=maxNumberOfJunctionsElement->getInt();

			//if(maxNumberOfJunctionsElement->findAttribute("NeighborOrder"))				
			//	fpptd.neighborOrder=maxNumberOfJunctionsElement->getAttributeAsUInt("NeighborOrder");

		}

		internalPlastParams[index] = fpptd;

		//inserting all the types to the set (duplicate are automatically eleminated) to figure out max value of type Id
		internalCellTypesSet.insert(type1);
		internalCellTypesSet.insert(type2);			
	}

	//Now that we know all the types used in the simulation we will find size of the plastParams
	vector<unsigned char> cellTypesVector(cellTypesSet.begin(),cellTypesSet.end());//coping set to the vector

	int size=0;
	int index ;
	if (cellTypesVector.size()){
		size= * max_element(cellTypesVector.begin(),cellTypesVector.end());
		size+=1;//if max element is e.g. 5 then size has to be 6 for an array to be properly allocated
	}

	plastParamsArray.clear();
	plastParamsArray.assign(size,vector<FocalPointPlasticityTrackerData>(size,FocalPointPlasticityTrackerData()));

	for(int i = 0 ; i < cellTypesVector.size() ; ++i)
		for(int j = 0 ; j < cellTypesVector.size() ; ++j){
			//cerr<<"cellTypesVector[i]="<<(int)cellTypesVector[i]<<endl;
			//cerr<<"cellTypesVector[j]="<<(int)cellTypesVector[j]<<endl;
			index = getIndex(cellTypesVector[i],cellTypesVector[j]);
			//cerr<<"index="<<index <<endl;

			plastParamsArray[cellTypesVector[i]][cellTypesVector[j]] = plastParams[index];
		}
		//initializing maxNumberOfJunctionsTotalVec based on plastParamsArray .maxNumberOfJunctionsTotalVec is indexed by cell type  	
		maxNumberOfJunctionsTotalVec.assign(size,0);
		for (int idx=0 ; idx<maxNumberOfJunctionsTotalVec.size() ; ++idx){
			int mNJ=0;
			for( int j =0 ; j < maxNumberOfJunctionsTotalVec.size() ; ++j){

				mNJ+=plastParamsArray[idx][j].maxNumberOfJunctions;
			}
			maxNumberOfJunctionsTotalVec[idx]=mNJ;
			cerr<<"maxNumberOfJunctions for type "<<idx<<" is "<<maxNumberOfJunctionsTotalVec[idx]<<endl;
		}

		//Now internal parameters
		//Now that we know all the types used in the simulation we will find size of the plastParams
		vector<unsigned char> internalCellTypesVector(internalCellTypesSet.begin(),internalCellTypesSet.end());//coping set to the vector

		size=0;
		if (internalCellTypesVector.size()){
			size= * max_element(internalCellTypesVector.begin(),internalCellTypesVector.end());
			size+=1;//if max element is e.g. 5 then size has to be 6 for an array to be properly allocated
		}

		internalPlastParamsArray.clear();
		internalPlastParamsArray.assign(size,vector<FocalPointPlasticityTrackerData>(size,FocalPointPlasticityTrackerData()));

		for(int i = 0 ; i < internalCellTypesVector.size() ; ++i)
			for(int j = 0 ; j < internalCellTypesVector.size() ; ++j){
				index = getIndex(internalCellTypesVector[i],internalCellTypesVector[j]);
				internalPlastParamsArray[internalCellTypesVector[i]][internalCellTypesVector[j]] = internalPlastParams[index];				
			}

			//initializing maxNumberOfJunctionsInternalTotalVec based on plastParamsArray .maxNumberOfJunctionsInternalTotalVec is indexed by cell type  	
			maxNumberOfJunctionsInternalTotalVec.assign(size,0);
			for (int idx=0 ; idx<maxNumberOfJunctionsInternalTotalVec.size() ; ++idx){
				int mNJ=0;
				for( int j =0 ; j < maxNumberOfJunctionsInternalTotalVec.size() ; ++j){

					mNJ+=internalPlastParamsArray[idx][j].maxNumberOfJunctions;
				}
				maxNumberOfJunctionsInternalTotalVec[idx]=mNJ;
				cerr<<"maxNumberOfInternalJunctions for type "<<idx<<" is "<<maxNumberOfJunctionsInternalTotalVec[idx]<<endl;
			}

			CC3DXMLElement * linkXMLElem = _xmlData->getFirstElement("LinkConstituentLaw");

			if( linkXMLElem  && linkXMLElem->findElement("Formula")){
				ASSERT_OR_THROW("CC3DML Error: Please change Formula tag to Expression tag inside LinkConstituentLaw element",false);
			}

			if (linkXMLElem){
				unsigned int maxNumberOfWorkNodes=pUtils->getMaxNumberOfWorkNodesPotts();
				eed.allocateSize(maxNumberOfWorkNodes);
				vector<string> variableNames;
				variableNames.push_back("Lambda");
				variableNames.push_back("Length");
				variableNames.push_back("TargetLength");
				

				eed.addVariables(variableNames.begin(),variableNames.end());
				eed.update(linkXMLElem);			
				constituentLawFcnPtr=&FocalPointPlasticityPlugin::customLinkConstituentLaw;
				
			}else{
				
				constituentLawFcnPtr=&FocalPointPlasticityPlugin::elasticLinkConstituentLaw;;
			}

			boundaryStrategy=BoundaryStrategy::getInstance();

}

double FocalPointPlasticityPlugin::potentialFunction(double _lambda,double _offset, double _targetDistance , double _distance){
	return _offset+_lambda*(_distance-_targetDistance)*(_distance-_targetDistance);
}

double FocalPointPlasticityPlugin::elasticLinkConstituentLaw(float _lambda,float _length,float _targetLength){

	return _lambda*(_length-_targetLength)*(_length-_targetLength);

}

double FocalPointPlasticityPlugin::customLinkConstituentLaw(float _lambda,float _length,float _targetLength){

		int currentWorkNodeNumber=pUtils->getCurrentWorkNodeNumber();	
		ExpressionEvaluator & ev=eed[currentWorkNodeNumber];
		double linkLaw=0.0;

		ev[0]=_lambda;
		ev[1]=_length;	
		ev[2]=_targetLength;	
		
		linkLaw=ev.eval();

		return linkLaw;

}

//not used
//double FocalPointPlasticityPlugin::diffEnergyLocal(float _deltaL,float _lBefore,const FocalPointPlasticityTrackerData * _plasticityTrackerData,const CellG *_cell,bool _useCluster){
//
//   float lambdaLocal=_plasticityTrackerData->lambdaDistance;
//   float targetDistanceLocal=_plasticityTrackerData->targetDistance;
//
//   if(_cell->volume>1){
//      return lambdaLocal*_deltaL*(2*(_lBefore-targetDistanceLocal)+_deltaL);
//   }else{//after spin flip oldCell will disappear so the only contribution from before spin flip i.e. -(l-l0)^2
//      return -lambdaLocal*(_lBefore-targetDistanceLocal)*(_lBefore-targetDistanceLocal);
//   }
//   	
//}

//double FocalPointPlasticityPlugin::diffEnergyByType(float _deltaL,float _lBefore,const FocalPointPlasticityTrackerData * _plasticityTrackerData,const CellG *_cell,bool _useCluster){
//   float lambdaDistanceLocal;
//   float targetDistanceLocal;
//   if(_useCluster){
//	   lambdaDistanceLocal=internalPlastParamsArray[_plasticityTrackerData->neighborAddress->type][_cell->type].lambdaDistance;
//	   targetDistanceLocal=internalPlastParamsArray[_plasticityTrackerData->neighborAddress->type][_cell->type].targetDistance;
//   }else{
//	   lambdaDistanceLocal=plastParamsArray[_plasticityTrackerData->neighborAddress->type][_cell->type].lambdaDistance;
//	   targetDistanceLocal=plastParamsArray[_plasticityTrackerData->neighborAddress->type][_cell->type].targetDistance;
//   }
//
//   if(_cell->volume>1){
//	   double diffEnergy=lambdaDistanceLocal*_deltaL*(2*(_lBefore-targetDistanceLocal)+_deltaL);
//
//	   //double diffEnergy=_plasticityTrackerData->lambdaDistance*_deltaL*(2*(_lBefore-_plasticityTrackerData->targetDistance)+_deltaL);
//	   //cerr<<"_plasticityTrackerData->targetDistance="<<_plasticityTrackerData->targetDistance<<endl;
//	   //cerr<<"_plasticityTrackerData->lambdaDistance="<<_plasticityTrackerData->lambdaDistance<<endl;
//	   //cerr<<"diff energy="<<diffEnergy<<endl;
//	   return diffEnergy;
//   }else{//after spin flip oldCell will disappear so the only contribution from before spin flip i.e. -(l-l0)^2
//		return -lambdaDistanceLocal*(_lBefore-targetDistanceLocal)*(_lBefore-targetDistanceLocal);
//      //return -_plasticityTrackerData->lambdaDistance*(_lBefore-_plasticityTrackerData->targetDistance)*(_lBefore-_plasticityTrackerData->targetDistance);
//   }
//}

double FocalPointPlasticityPlugin::diffEnergyLocal(float _deltaL,float _lBefore,const FocalPointPlasticityTrackerData * _plasticityTrackerData,const CellG *_cell,bool _useCluster){

	float lambdaLocal=_plasticityTrackerData->lambdaDistance;
	float targetDistanceLocal=_plasticityTrackerData->targetDistance;

	if(_cell->volume>1){

		return (this->*constituentLawFcnPtr)(lambdaLocal,_lBefore+_deltaL,targetDistanceLocal)-(this->*constituentLawFcnPtr)(lambdaLocal,_lBefore,targetDistanceLocal);
	}else{//after spin flip oldCell will disappear so the only contribution from before spin flip i.e. -(l-l0)^2
		return -(this->*constituentLawFcnPtr)(lambdaLocal,_lBefore,targetDistanceLocal);
	}

}

//not used
double FocalPointPlasticityPlugin::diffEnergyGlobal(float _deltaL,float _lBefore,const FocalPointPlasticityTrackerData * _plasticityTrackerData,const CellG *_cell,bool _useCluster){

	if(_cell->volume>1){
		return lambda*_deltaL*(2*(_lBefore-targetDistance)+_deltaL);
	}else{//after spin flip oldCell will disappear so the only contribution from before spin flip i.e. -(l-l0)^2
		return -lambda*(_lBefore-targetDistance)*(_lBefore-targetDistance);
	}

}

double FocalPointPlasticityPlugin::diffEnergyByType(float _deltaL,float _lBefore,const FocalPointPlasticityTrackerData * _plasticityTrackerData,const CellG *_cell,bool _useCluster){
	float lambdaDistanceLocal;
	float targetDistanceLocal;
	if(_useCluster){
		lambdaDistanceLocal=internalPlastParamsArray[_plasticityTrackerData->neighborAddress->type][_cell->type].lambdaDistance;
		targetDistanceLocal=internalPlastParamsArray[_plasticityTrackerData->neighborAddress->type][_cell->type].targetDistance;
	}else{
		lambdaDistanceLocal=plastParamsArray[_plasticityTrackerData->neighborAddress->type][_cell->type].lambdaDistance;
		targetDistanceLocal=plastParamsArray[_plasticityTrackerData->neighborAddress->type][_cell->type].targetDistance;
	}

	if(_cell->volume>1){
		return (this->*constituentLawFcnPtr)(lambdaDistanceLocal,_lBefore+_deltaL,targetDistanceLocal)-(this->*constituentLawFcnPtr)(lambdaDistanceLocal,_lBefore,targetDistanceLocal);
	}else{//after spin flip oldCell will disappear so the only contribution from before spin flip i.e. -(l-l0)^2
		return -(this->*constituentLawFcnPtr)(lambdaDistanceLocal,_lBefore,targetDistanceLocal);
	}
}

void FocalPointPlasticityPlugin::insertFPPData(CellG * _cell,FocalPointPlasticityTrackerData * _fpptd){
	FocalPointPlasticityTrackerData fpptd(* _fpptd);
	focalPointPlasticityTrackerAccessor.get(_cell->extraAttribPtr)->focalPointPlasticityNeighbors.insert(fpptd);
}

void FocalPointPlasticityPlugin::insertInternalFPPData(CellG * _cell,FocalPointPlasticityTrackerData * _fpptd){
	FocalPointPlasticityTrackerData fpptd(* _fpptd);
	focalPointPlasticityTrackerAccessor.get(_cell->extraAttribPtr)->internalFocalPointPlasticityNeighbors.insert(fpptd);
}


void FocalPointPlasticityPlugin::insertAnchorFPPData(CellG * _cell,FocalPointPlasticityTrackerData * _fpptd){
	FocalPointPlasticityTrackerData fpptd(* _fpptd);
	focalPointPlasticityTrackerAccessor.get(_cell->extraAttribPtr)->anchors.insert(fpptd);
}


std::vector<FocalPointPlasticityTrackerData> FocalPointPlasticityPlugin::getFPPDataVec(CellG * _cell){
	std::vector<FocalPointPlasticityTrackerData> dataVec;
	for each(FocalPointPlasticityLink* link in linkInv.getCellLinkList(_cell))
		dataVec.push_back(link->getFPPTrackerData(_cell));
	return dataVec;
}

std::vector<FocalPointPlasticityTrackerData> FocalPointPlasticityPlugin::getInternalFPPDataVec(CellG * _cell){
	std::vector<FocalPointPlasticityTrackerData> dataVec;
	for each(FocalPointPlasticityInternalLink* link in linkInvInternal.getCellLinkList(_cell))
		dataVec.push_back(link->getFPPTrackerData(_cell));
	return dataVec;
}

std::vector<FocalPointPlasticityTrackerData> FocalPointPlasticityPlugin::getAnchorFPPDataVec(CellG * _cell){
	std::vector<FocalPointPlasticityTrackerData> dataVec;
	for each (FocalPointPlasticityAnchor* link in linkInvAnchor.getCellLinkList(_cell))
		dataVec.push_back(link->getFPPTrackerData(_cell));
	return dataVec;
}

double FocalPointPlasticityPlugin::tryAddingNewJunction(const Point3D &pt,const CellG *newCell) {

	int currentWorkNodeNumber=pUtils->getCurrentWorkNodeNumber();	
	short &  newJunctionInitiatedFlag = newJunctionInitiatedFlagVec[currentWorkNodeNumber];
	CellG * & newNeighbor=newNeighborVec[currentWorkNodeNumber];

	if (((int)plastParamsArray.size())-1<newCell->type){ //the newCell type is not listed by the user
		newJunctionInitiatedFlag=false;
		return 0.0;
	}

	//check if new cell can accept new junctions
	if(focalPointPlasticityTrackerAccessor.get(newCell->extraAttribPtr)->focalPointPlasticityNeighbors.size()>=maxNumberOfJunctionsTotalVec[newCell->type]){
		newJunctionInitiatedFlag=false;
		return 0.0;

	}

	boundaryStrategy=BoundaryStrategy::getInstance();

	int maxNeighborIndexLocal=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(neighborOrder);
	Neighbor neighbor;
	CellG * nCell;
	WatchableField3D<CellG *> *fieldG =(WatchableField3D<CellG *> *) potts->getCellFieldG();
	//visit point neighbors of newCell and see if within of specified range there is another cell with which newCell can make a junction

	// Randomly shuffle neighbor selection
	std::vector<unsigned int> randNeighborIndex = std::vector<unsigned int>(maxNeighborIndexLocal + 1, 0);
	for (unsigned int nIdx = 0; nIdx <= maxNeighborIndexLocal; ++nIdx) randNeighborIndex[nIdx] = nIdx;
	std::random_shuffle(randNeighborIndex.begin(), randNeighborIndex.end());

	for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndexLocal ; ++nIdx ){
		neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt), randNeighborIndex[nIdx]);
		if(!neighbor.distance){
			//if distance is 0 then the neighbor returned is invalid
			continue;
		}
		nCell=fieldG->get(neighbor.pt);

		if (!nCell) //no junctions with medium
			continue;


		if (nCell==newCell || nCell->clusterId==newCell->clusterId)	// make sure that newCell and nCell are different and belong to different clusters
			continue;

		//check if type of neighbor cell is listed by the user
		if(((int)plastParamsArray.size())-1<nCell->type || plastParamsArray[newCell->type][nCell->type].maxNumberOfJunctions==0){	
			continue;
		}

		// check if neighbor cell can accept another junction
		set<FocalPointPlasticityTrackerData> &nCellFPPTD=focalPointPlasticityTrackerAccessor.get(nCell->extraAttribPtr)->focalPointPlasticityNeighbors;
		int currentNumberOfJunctionsNCell = count_if(nCellFPPTD.begin(),nCellFPPTD.end(),FocalPointPlasticityJunctionCounter(newCell->type)); //this will count number of junctions between nCell and cells of same type as newCell
		if( currentNumberOfJunctionsNCell >= plastParamsArray[newCell->type][nCell->type].maxNumberOfJunctions ){
			continue;
		}


		// check if new cell can accept another junction
		set<FocalPointPlasticityTrackerData> &newCellFPPTD=focalPointPlasticityTrackerAccessor.get(newCell->extraAttribPtr)->focalPointPlasticityNeighbors;
		int currentNumberOfJunctionsNewCell = count_if(newCellFPPTD.begin(),newCellFPPTD.end(),FocalPointPlasticityJunctionCounter(nCell->type)); //this will count number of junctions between newCell and cells of same type as nCell
		if(currentNumberOfJunctionsNewCell>=plastParamsArray[newCell->type][nCell->type].maxNumberOfJunctions){
			continue;
		}

		//check if nCell has a junction with newCell                
		set<FocalPointPlasticityTrackerData>::iterator sitr=
			focalPointPlasticityTrackerAccessor.get(newCell->extraAttribPtr)->focalPointPlasticityNeighbors.find(FocalPointPlasticityTrackerData(nCell));
		if(sitr==focalPointPlasticityTrackerAccessor.get(newCell->extraAttribPtr)->focalPointPlasticityNeighbors.end()){
			//new connection allowed			
			newJunctionInitiatedFlag=true;
			newNeighbor=nCell;
			break; 
		}
	}

	if(newJunctionInitiatedFlag){
		return plastParamsArray[newCell->type][newNeighbor->type].activationEnergy;
	}	else{
		return 0.0;
	}
}

double FocalPointPlasticityPlugin::tryAddingNewJunctionWithinCluster(const Point3D &pt,const CellG *newCell) {
	int currentWorkNodeNumber=pUtils->getCurrentWorkNodeNumber();	
	short &  newJunctionInitiatedFlagWithinCluster = newJunctionInitiatedFlagWithinClusterVec[currentWorkNodeNumber];
	CellG * & newNeighbor=newNeighborVec[currentWorkNodeNumber];

	if (((int)internalPlastParamsArray.size())-1<newCell->type){ //the newCell type is not listed by the user    
		newJunctionInitiatedFlagWithinCluster=false;
		return 0.0;
	}

	//check if new cell can accept new junctions
	if(focalPointPlasticityTrackerAccessor.get(newCell->extraAttribPtr)->internalFocalPointPlasticityNeighbors.size()>=maxNumberOfJunctionsInternalTotalVec[newCell->type]){
		newJunctionInitiatedFlagWithinCluster=false;
		return 0.0;
	}

	boundaryStrategy=BoundaryStrategy::getInstance();
	int maxNeighborIndexLocal=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(neighborOrder);
	Neighbor neighbor;
	CellG * nCell;
	WatchableField3D<CellG *> *fieldG =(WatchableField3D<CellG *> *) potts->getCellFieldG();
	//visit point neighbors of newCell and see if within of specified range there is another cell with which newCell can make a junction

	// Randomly shuffle neighbor selection
	std::vector<unsigned int> randNeighborIndex = std::vector<unsigned int>(maxNeighborIndexLocal + 1, 0);
	for (unsigned int nIdx = 0; nIdx <= maxNeighborIndexLocal; ++nIdx) randNeighborIndex[nIdx] = nIdx;
	std::random_shuffle(randNeighborIndex.begin(), randNeighborIndex.end());

	for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndexLocal ; ++nIdx ){
		neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt), randNeighborIndex[nIdx]);
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
		if(((int)internalPlastParamsArray.size())-1<nCell->type || maxNumberOfJunctionsInternalTotalVec[newCell->type]==0){	
			continue;
		}


		// check if neighbor cell can accept another junction

		set<FocalPointPlasticityTrackerData> &nCellFPPTD=focalPointPlasticityTrackerAccessor.get(nCell->extraAttribPtr)->internalFocalPointPlasticityNeighbors;
		int currentNumberOfJunctionsNCell = count_if(nCellFPPTD.begin(),nCellFPPTD.end(),FocalPointPlasticityJunctionCounter(newCell->type)); //this will count number of junctions between nCell and cells of same type as newCell
		if( currentNumberOfJunctionsNCell >= internalPlastParamsArray[newCell->type][nCell->type].maxNumberOfJunctions ){
			continue;
		}

		// check if newCell can accept another junction

		set<FocalPointPlasticityTrackerData> &newCellFPPTD=focalPointPlasticityTrackerAccessor.get(newCell->extraAttribPtr)->internalFocalPointPlasticityNeighbors;
		int currentNumberOfJunctionsNewCell = count_if(newCellFPPTD.begin(),newCellFPPTD.end(),FocalPointPlasticityJunctionCounter(nCell->type)); //this will count number of junctions between newCell and cells of same type as nCell
		if(currentNumberOfJunctionsNewCell>=internalPlastParamsArray[newCell->type][nCell->type].maxNumberOfJunctions){
			continue;
		}

		//check if nCell has has a junction with newCell                
		set<FocalPointPlasticityTrackerData>::iterator sitr=
			focalPointPlasticityTrackerAccessor.get(newCell->extraAttribPtr)->internalFocalPointPlasticityNeighbors.find(FocalPointPlasticityTrackerData(nCell));
		if(sitr==focalPointPlasticityTrackerAccessor.get(newCell->extraAttribPtr)->internalFocalPointPlasticityNeighbors.end()){
			//new connection allowed
			newJunctionInitiatedFlagWithinCluster=true;
			newNeighbor=nCell;
			break; 
		}
	}


	if(newJunctionInitiatedFlagWithinCluster){
		return internalPlastParamsArray[newCell->type][newNeighbor->type].activationEnergy;
	}	else{
		return 0.0;
	}

}

double FocalPointPlasticityPlugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {

	//This plugin will not work properly with periodic boundary conditions. If necessary I can fix it
	//cerr<<"THIS IS pt="<<pt<<endl;
	if (newCell==oldCell) //this may happen if you are trying to assign same cell to one pixel twice 
		return 0.0;


	int currentWorkNodeNumber=pUtils->getCurrentWorkNodeNumber();	
	short &  newJunctionInitiatedFlag = newJunctionInitiatedFlagVec[currentWorkNodeNumber];
	short &  newJunctionInitiatedFlagWithinCluster = newJunctionInitiatedFlagWithinClusterVec[currentWorkNodeNumber];
	CellG * & newNeighbor=newNeighborVec[currentWorkNodeNumber];

	double energy=0.0;
	WatchableField3D<CellG *> *fieldG =(WatchableField3D<CellG *> *) potts->getCellFieldG();

	Neighbor neighbor;
	Neighbor neighborOfNeighbor;
	CellG * nCell;

	newJunctionInitiatedFlag=false;
	newJunctionInitiatedFlagWithinCluster=false;

	newNeighbor=0;

	//check if we need to create new junctions only new cell can initiate junctions
	if(newCell){
		//cerr<<"trying adding new cell within a cluster"<<endl;
		double activationEnergy=tryAddingNewJunctionWithinCluster(pt,newCell);
		if(newJunctionInitiatedFlagWithinCluster){
			//cerr<<"GOT NEW JUNCTION"<<endl;

			//exit(0);
			energy+=activationEnergy;

			return energy;
		}
	}

	if(newCell){
		double activationEnergy=tryAddingNewJunction(pt,newCell);		
		if(newJunctionInitiatedFlag){
			//cerr<<"GOT NEW INTERNAL JUNCTION"<<endl;

			//exit(0);
			energy+=activationEnergy;

			return energy;
		}
	}
	//cerr<<"GOT HERE"<<endl;


	Coordinates3D<double> centroidOldAfter;
	Coordinates3D<double> centroidNewAfter;
	Coordinates3D<float> centMassOldAfter;
	Coordinates3D<float> centMassNewAfter;
	Coordinates3D<float> centMassOldBefore;
	Coordinates3D<float> centMassNewBefore;


	//    cerr<<"fieldDim="<<fieldDim<<endl;
	if(oldCell){
		centMassOldBefore.XRef()=oldCell->xCM/(float)oldCell->volume;
		centMassOldBefore.YRef()=oldCell->yCM/(float)oldCell->volume;
		centMassOldBefore.ZRef()=oldCell->zCM/(float)oldCell->volume;

		if(oldCell->volume>1){
			//cerr<<"THIS IS oldCellVolume="<<oldCell->volume<<endl;
			//cerr<<" FPP boundaryStrategy="<<boundaryStrategy<<endl;
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
		centMassNewBefore.XRef()=newCell->xCM/(float)newCell->volume;
		centMassNewBefore.YRef()=newCell->yCM/(float)newCell->volume;
		centMassNewBefore.ZRef()=newCell->zCM/(float)newCell->volume;

		centroidNewAfter=precalculateCentroid(pt, newCell, 1,fieldDim,boundaryStrategy);
		centMassNewAfter.XRef()=centroidNewAfter.X()/(float)(newCell->volume+1);
		centMassNewAfter.YRef()=centroidNewAfter.Y()/(float)(newCell->volume+1);
		centMassNewAfter.ZRef()=centroidNewAfter.Z()/(float)(newCell->volume+1);
	}

	//will loop over neighbors of the oldCell and calculate Plasticity energy

	std::vector<float> anchorPoint;

	float deltaL;
	float lBefore;
	float oldVol;
	float newVol;
	float nCellVol;
	//cerr<<"energy before old cell section="<<energy<<endl;
	if(oldCell){
		//cerr<<"energy for old cell section"<<endl;

		for each (FocalPointPlasticityLink* link in linkInv.getCellLinkList(const_cast<CellG*>(oldCell))) {
			nCell = link->getOtherCell(oldCell);

			if(nCell!=newCell){
				lBefore=distInvariantCM(centMassOldBefore.X(),centMassOldBefore.Y(),centMassOldBefore.Z(),nCell->xCM/nCellVol,nCell->yCM/nCellVol,nCell->zCM/nCellVol,fieldDim,boundaryStrategy);
				deltaL=
					distInvariantCM(centMassOldAfter.X(),centMassOldAfter.Y(),centMassOldAfter.Z(),nCell->xCM/nCellVol,nCell->yCM/nCellVol,nCell->zCM/nCellVol,fieldDim,boundaryStrategy)
					-lBefore;
			}else{
				lBefore=distInvariantCM(centMassOldBefore.X(),centMassOldBefore.Y(),centMassOldBefore.Z(),centMassNewBefore.X(),centMassNewBefore.Y(),centMassNewBefore.Z(),fieldDim,boundaryStrategy);
				deltaL=
					distInvariantCM(centMassOldAfter.X(),centMassOldAfter.Y(),centMassOldAfter.Z(),centMassNewAfter.X(),centMassNewAfter.Y(),centMassNewAfter.Z(),fieldDim,boundaryStrategy)
					-lBefore;
			}

			energy+=(this->*diffEnergyFcnPtr)(deltaL,lBefore,&link->getFPPTrackerData(const_cast<CellG*>(oldCell)),oldCell,false);
		}

		//go over compartments
		for each (FocalPointPlasticityInternalLink* link in linkInvInternal.getCellLinkList(const_cast<CellG*>(oldCell))) {
			nCell = link->getOtherCell(oldCell);
			nCellVol=nCell->volume;

			if(nCell!=newCell){
				lBefore=distInvariantCM(centMassOldBefore.X(),centMassOldBefore.Y(),centMassOldBefore.Z(),nCell->xCM/nCellVol,nCell->yCM/nCellVol,nCell->zCM/nCellVol,fieldDim,boundaryStrategy);
				deltaL=
					distInvariantCM(centMassOldAfter.X(),centMassOldAfter.Y(),centMassOldAfter.Z(),nCell->xCM/nCellVol,nCell->yCM/nCellVol,nCell->zCM/nCellVol,fieldDim,boundaryStrategy)
					-lBefore;
			}else{
				lBefore=distInvariantCM(centMassOldBefore.X(),centMassOldBefore.Y(),centMassOldBefore.Z(),centMassNewBefore.X(),centMassNewBefore.Y(),centMassNewBefore.Z(),fieldDim,boundaryStrategy);
				deltaL=
					distInvariantCM(centMassOldAfter.X(),centMassOldAfter.Y(),centMassOldAfter.Z(),centMassNewAfter.X(),centMassNewAfter.Y(),centMassNewAfter.Z(),fieldDim,boundaryStrategy)
					-lBefore;
			}
			//cerr<<"deltaL="<<deltaL<<" lBefore="<<lBefore<<" lambda="<<sitr->lambdaDistance<<endl;
			double clusterOldCellEnergy=(this->*diffEnergyFcnPtr)(deltaL,lBefore, &link->getFPPTrackerData(const_cast<CellG*>(oldCell)),oldCell,true);
			//cerr<<"clusterOldCellEnergy="<<clusterOldCellEnergy<<endl;
			energy+=clusterOldCellEnergy;
		}

		//go over anchors
		for each (FocalPointPlasticityAnchor* link in linkInvAnchor.getCellLinkList(const_cast<CellG*>(oldCell))) {
			anchorPoint = link->getAnchorPoint();
			lBefore=distInvariantCM(centMassOldBefore.X(),centMassOldBefore.Y(),centMassOldBefore.Z(),anchorPoint[0],anchorPoint[1],anchorPoint[2],fieldDim,boundaryStrategy);
			deltaL=
				distInvariantCM(centMassOldAfter.X(),centMassOldAfter.Y(),centMassOldAfter.Z(),anchorPoint[0],anchorPoint[1],anchorPoint[2],fieldDim,boundaryStrategy)
				-lBefore;
			energy+=(this->*diffEnergyFcnPtr)(deltaL,lBefore, &link->getFPPTrackerData(const_cast<CellG*>(oldCell)),oldCell,false);
		}

	}

	//cerr<<"energy before new cell section="<<energy<<endl;
	if(newCell){
		//cerr<<"energy for new cell section"<<endl;
		//cerr<<"energy before new section starts="<<energy<<endl;
		
		for each (FocalPointPlasticityLink* link in linkInv.getCellLinkList(const_cast<CellG*>(newCell))) {
			nCell = link->getOtherCell(newCell);
			nCellVol = nCell->volume;

			if (nCell != oldCell) {
				lBefore = distInvariantCM(centMassNewBefore.X(), centMassNewBefore.Y(), centMassNewBefore.Z(), nCell->xCM / nCellVol, nCell->yCM / nCellVol, nCell->zCM / nCellVol, fieldDim, boundaryStrategy);
				deltaL =
					distInvariantCM(centMassNewAfter.X(), centMassNewAfter.Y(), centMassNewAfter.Z(), nCell->xCM / nCellVol, nCell->yCM / nCellVol, nCell->zCM / nCellVol, fieldDim, boundaryStrategy)
					- lBefore;

				double newDeltaEnergy = (this->*diffEnergyFcnPtr)(deltaL, lBefore, &link->getFPPTrackerData(const_cast<CellG*>(newCell)), newCell, false);
				//cerr<<"newDeltaEnergy="<<newDeltaEnergy<<endl;
				energy += newDeltaEnergy;

			}
			else {
				// this was already taken into account in the oldCell secion - we need to avoid double counting
			}
		}

		//go ever compartments
		for each (FocalPointPlasticityInternalLink* link in linkInvInternal.getCellLinkList(const_cast<CellG*>(newCell))) {
			nCell = link->getOtherCell(newCell);
			nCellVol=nCell->volume;

			if(nCell!=oldCell){
				lBefore=distInvariantCM(centMassNewBefore.X(),centMassNewBefore.Y(),centMassNewBefore.Z(),nCell->xCM/nCellVol,nCell->yCM/nCellVol,nCell->zCM/nCellVol,fieldDim,boundaryStrategy);
				deltaL=
					distInvariantCM(centMassNewAfter.X(),centMassNewAfter.Y(),centMassNewAfter.Z(),nCell->xCM/nCellVol,nCell->yCM/nCellVol,nCell->zCM/nCellVol,fieldDim,boundaryStrategy)
					-lBefore;
				energy+=(this->*diffEnergyFcnPtr)(deltaL,lBefore, &link->getFPPTrackerData(const_cast<CellG*>(newCell)),newCell,true);
			}
			else{
				// this was already taken into account in the oldCell secion - we need to avoid double counting
			} 
			//////energy+=(this->*diffEnergyFcnPtr)(deltaL,lBefore,&(*sitr),newCell,true);
		}

		//go over anchors
		for each (FocalPointPlasticityAnchor* link in linkInvAnchor.getCellLinkList(const_cast<CellG*>(newCell))) {
			anchorPoint = link->getAnchorPoint();
			lBefore=distInvariantCM(centMassNewBefore.X(),centMassNewBefore.Y(),centMassNewBefore.Z(),anchorPoint[0],anchorPoint[1],anchorPoint[2],fieldDim,boundaryStrategy);
			deltaL=
				distInvariantCM(centMassNewAfter.X(),centMassNewAfter.Y(),centMassNewAfter.Z(),anchorPoint[0],anchorPoint[1],anchorPoint[2],fieldDim,boundaryStrategy)
				-lBefore;

			energy+=(this->*diffEnergyFcnPtr)(deltaL,lBefore, &link->getFPPTrackerData(const_cast<CellG*>(newCell)),newCell,false);
		}

	}
	//cerr<<"pt="<<pt<<" DELTA E="<<energy<<endl;
	return energy;
}


void FocalPointPlasticityPlugin::deleteFocalPointPlasticityLink(CellG * _cell1,CellG * _cell2){
	std::set<FocalPointPlasticityTrackerData>::iterator sitr;
	std::set<FocalPointPlasticityTrackerData> & plastNeighbors1=focalPointPlasticityTrackerAccessor.get(_cell1->extraAttribPtr)->focalPointPlasticityNeighbors;
	plastNeighbors1.erase(FocalPointPlasticityTrackerData(_cell2));

	std::set<FocalPointPlasticityTrackerData> & plastNeighbors2=focalPointPlasticityTrackerAccessor.get(_cell2->extraAttribPtr)->focalPointPlasticityNeighbors;
	plastNeighbors2.erase(FocalPointPlasticityTrackerData(_cell1));

	FocalPointPlasticityLink* link = linkInv.getLinkByCells(_cell1, _cell2);
	if (link) linkInv.removeFromInventory(link);

}

void FocalPointPlasticityPlugin::deleteInternalFocalPointPlasticityLink(CellG * _cell1,CellG * _cell2){
	std::set<FocalPointPlasticityTrackerData> & internalPlastNeighbors1=focalPointPlasticityTrackerAccessor.get(_cell1->extraAttribPtr)->internalFocalPointPlasticityNeighbors;
	internalPlastNeighbors1.erase(FocalPointPlasticityTrackerData(_cell2));

	std::set<FocalPointPlasticityTrackerData> & internalPlastNeighbors2=focalPointPlasticityTrackerAccessor.get(_cell2->extraAttribPtr)->internalFocalPointPlasticityNeighbors;
	internalPlastNeighbors2.erase(FocalPointPlasticityTrackerData(_cell1));

	FocalPointPlasticityInternalLink* link = linkInvInternal.getLinkByCells(_cell1, _cell2);
	if (link) linkInvInternal.removeFromInventory(link);

}

void FocalPointPlasticityPlugin::createFocalPointPlasticityLink(CellG * _cell1,CellG * _cell2,double _lambda, double _targetDistance,double _maxDistance){
//	FocalPointPlasticityTrackerData fpptd1=plastParamsArray[_cell1->type][_cell2->type];

	FocalPointPlasticityTrackerData fpptd1;
	fpptd1.targetDistance=_targetDistance;
	fpptd1.lambdaDistance=_lambda;
	fpptd1.maxDistance=_maxDistance;			
	fpptd1.neighborAddress=_cell2;
	fpptd1.isInitiator = true;
	fpptd1.initMCS = sim->getStep();

	focalPointPlasticityTrackerAccessor.get(_cell1->extraAttribPtr)->focalPointPlasticityNeighbors.insert(fpptd1);


//	FocalPointPlasticityTrackerData fpptd2=plastParamsArray[_cell2->type][_cell1->type];
    FocalPointPlasticityTrackerData fpptd2;
	fpptd2.targetDistance=_targetDistance;
	fpptd2.lambdaDistance=_lambda;
	fpptd2.maxDistance=_maxDistance;			
	fpptd2.neighborAddress=_cell1;
	fpptd2.isInitiator = false;
	fpptd2.initMCS = sim->getStep();

	focalPointPlasticityTrackerAccessor.get(_cell2->extraAttribPtr)->focalPointPlasticityNeighbors.insert(fpptd2);

	FocalPointPlasticityLink link = FocalPointPlasticityLink(_cell1, _cell2, potts, fpptd1);
	linkInv.addToInventory(&link);

}

void FocalPointPlasticityPlugin::createInternalFocalPointPlasticityLink(CellG * _cell1,CellG * _cell2,double _lambda, double _targetDistance,double _maxDistance){
//	FocalPointPlasticityTrackerData fpptd1=internalPlastParamsArray[_cell1->type][_cell2->type];
    FocalPointPlasticityTrackerData fpptd1;
	fpptd1.targetDistance=_targetDistance;
	fpptd1.lambdaDistance=_lambda;
	fpptd1.maxDistance=_maxDistance;			
	fpptd1.neighborAddress=_cell2;
	fpptd1.isInitiator = true;
	fpptd1.initMCS = sim->getStep();

	focalPointPlasticityTrackerAccessor.get(_cell1->extraAttribPtr)->internalFocalPointPlasticityNeighbors.insert(fpptd1);


//	FocalPointPlasticityTrackerData fpptd2=internalPlastParamsArray[_cell2->type][_cell1->type];
    FocalPointPlasticityTrackerData fpptd2;
	fpptd2.targetDistance=_targetDistance;
	fpptd2.lambdaDistance=_lambda;
	fpptd2.maxDistance=_maxDistance;			
	fpptd2.neighborAddress=_cell1;
	fpptd2.isInitiator = false;
	fpptd2.initMCS = sim->getStep();

	focalPointPlasticityTrackerAccessor.get(_cell2->extraAttribPtr)->internalFocalPointPlasticityNeighbors.insert(fpptd2);

	FocalPointPlasticityInternalLink link = FocalPointPlasticityInternalLink(_cell1, _cell2, potts, fpptd1);
	linkInvInternal.addToInventory(&link);

}


void FocalPointPlasticityPlugin::field3DChange(const Point3D &pt, CellG *newCell,CellG *oldCell){

	int currentWorkNodeNumber=pUtils->getCurrentWorkNodeNumber();	
	short &  newJunctionInitiatedFlag = newJunctionInitiatedFlagVec[currentWorkNodeNumber];
	short &  newJunctionInitiatedFlagWithinCluster = newJunctionInitiatedFlagWithinClusterVec[currentWorkNodeNumber];
	CellG * & newNeighbor=newNeighborVec[currentWorkNodeNumber];

	if(oldCell && oldCell->volume==0){

		//cerr<<"\t\t\t oldCell.id="<<oldCell->id<<" is to be removed"<<endl;

		for each (FocalPointPlasticityLink* link in linkInv.getCellLinkList(oldCell))
			deleteFocalPointPlasticityLink(oldCell, link->getOtherCell(oldCell));

		//go over compartments
		for each (FocalPointPlasticityInternalLink* link in linkInvInternal.getCellLinkList(oldCell))
			deleteInternalFocalPointPlasticityLink(oldCell, link->getOtherCell(oldCell));

		//go over anchors
		for each (FocalPointPlasticityAnchor* link in linkInvAnchor.getCellLinkList(oldCell))
			deleteAnchor(oldCell, link->getAnchorId());
	}

	//if(oldCell && oldCell->id==1082){
	//	cerr<<"\t\t oldCell 1082 volume="<<oldCell->volume<<endl;
	//	cerr<<"newJunctionInitiatedFlag="<<newJunctionInitiatedFlag<<endl;
	//	cerr<<"newJunctionInitiatedFlagWithinCluster="<<newJunctionInitiatedFlagWithinCluster<<endl;
	//}
	//if(newCell && newCell->id==1082){
	//	cerr<<"\t\t newCell 1082 volume="<<newCell->volume<<endl;
	//}

	//because newJunctionInitiatedFlag is in principle "global" variable changing lattice configuration from python e.g. delete cell
	// can cause newJunctionInitiatedFlag be still after last change and thus falsly indicate new junction even when cell is e.g. Medium



	if(newJunctionInitiatedFlag && newCell){ 

		// we reset the flags here to avoid  keeping true value in the "global" class-wide variable- this might have ide effects
		newJunctionInitiatedFlag = false;

		//if(newCell->id==1082 ||newNeighbor->id==1082  ){
		//cerr<<"newJunctionInitiatedFlag="<<newJunctionInitiatedFlag<<" newCell="<<newCell->id<<endl;
		//cerr<<"newJunctionInitiatedFlag="<<newJunctionInitiatedFlag<<" newNeighborCell="<<newNeighbor->id<<endl;
		//}
		double xCMNew=newCell->xCM/float(newCell->volume);
		double yCMNew=newCell->yCM/float(newCell->volume);
		double zCMNew=newCell->zCM/float(newCell->volume);

		double xCMNeighbor=newNeighbor->xCM/float(newNeighbor->volume);
		double yCMNeighbor=newNeighbor->yCM/float(newNeighbor->volume);
		double zCMNeighbor=newNeighbor->zCM/float(newNeighbor->volume);

		double distance=distInvariantCM(xCMNew,yCMNew,zCMNew,xCMNeighbor,yCMNeighbor,zCMNeighbor,fieldDim,boundaryStrategy);
		//double distance=dist(xCMNew,yCMNew,zCMNew,xCMNeighbor,yCMNeighbor,zCMNeighbor);

		//if (plasticityTypes.size()==0||(plasticityTypes.find(newNeighbor->type)!=plasticityTypes.end() && plasticityTypes.find(newCell->type)!=plasticityTypes.end())){

		if (functionType==BYCELLTYPE || functionType==BYCELLID){
			//cerr<<"adding external junction between "<<newCell<<" and "<<newNeighbor<<endl;
			FocalPointPlasticityTrackerData fpptd=plastParamsArray[newCell->type][newNeighbor->type];
			createFocalPointPlasticityLink(newCell, newNeighbor, fpptd.lambdaDistance, fpptd.targetDistance, fpptd.maxDistance);

		}else if (functionType==GLOBAL){
			FocalPointPlasticityTrackerData fpptd = FocalPointPlasticityTrackerData(newNeighbor, lambda, 0.9*distance);
			createFocalPointPlasticityLink(newCell, newNeighbor, fpptd.lambdaDistance, fpptd.targetDistance, fpptd.maxDistance);

		}
		//}
		return;
	}

	//because newJunctionInitiatedFlag is in principle "global" variable changing lattice configuration from python e.g. delete cell
	// can cause newJunctionInitiatedFlag be still after last change and thus falsly indicate new junction even when cell is e.g. Medium

	if (newJunctionInitiatedFlagWithinCluster && newCell){

		// we reset the flags here to avoid  keeping true value in the "global" class-wide variable- this might have ide effects		
		newJunctionInitiatedFlagWithinCluster = false;

		//if(newCell->id==1082 ||newNeighbor->id==1082  ){
		//cerr<<"newJunctionInitiatedFlagWithinCluster="<<newJunctionInitiatedFlagWithinCluster<<" newCell="<<newCell->id<<endl;
		//cerr<<"newJunctionInitiatedFlagWithinCluster="<<newJunctionInitiatedFlagWithinCluster<<" newNeighborCell="<<newNeighbor->id<<endl;
		//}
		double xCMNew=newCell->xCM/float(newCell->volume);
		double yCMNew=newCell->yCM/float(newCell->volume);
		double zCMNew=newCell->zCM/float(newCell->volume);

		double xCMNeighbor=newNeighbor->xCM/float(newNeighbor->volume);
		double yCMNeighbor=newNeighbor->yCM/float(newNeighbor->volume);
		double zCMNeighbor=newNeighbor->zCM/float(newNeighbor->volume);

		double distance=distInvariantCM(xCMNew,yCMNew,zCMNew,xCMNeighbor,yCMNeighbor,zCMNeighbor,fieldDim,boundaryStrategy);
		//double distance=dist(xCMNew,yCMNew,zCMNew,xCMNeighbor,yCMNeighbor,zCMNeighbor);

		//if (plasticityTypes.size()==0||(plasticityTypes.find(newNeighbor->type)!=plasticityTypes.end() && plasticityTypes.find(newCell->type)!=plasticityTypes.end())){
		if (functionType==BYCELLTYPE || functionType==BYCELLID){
			//cerr<<"adding internal junction between "<<newCell<<" and "<<newNeighbor<<endl;
			FocalPointPlasticityTrackerData fpptd=internalPlastParamsArray[newCell->type][newNeighbor->type];
			createInternalFocalPointPlasticityLink(newCell, newNeighbor, fpptd.lambdaDistance, fpptd.targetDistance, fpptd.maxDistance);

		}else if (functionType==GLOBAL){
			FocalPointPlasticityTrackerData fpptd = FocalPointPlasticityTrackerData(newNeighbor, lambda, 0.9*distance);
			createInternalFocalPointPlasticityLink(newCell, newNeighbor, fpptd.lambdaDistance, fpptd.targetDistance, fpptd.maxDistance);

		}
		//}
		return;
	}

	// we reset the flags here to avoid  keeping true value in the "global" class-wide variable- this might have ide effects
	newJunctionInitiatedFlag = false;
	newJunctionInitiatedFlagWithinCluster = false;

	CellG* nCell;

	if(newCell){
		double xCMNew=newCell->xCM/float(newCell->volume);
		double yCMNew=newCell->yCM/float(newCell->volume);
		double zCMNew=newCell->zCM/float(newCell->volume);

		//list<CellG *> toBeRemovedNeighborsOfNewCell;

		//we remove only one cell at a time even though we could do it for many cells 
		bool linkRemoved = false;
		std::vector<float> anchorPoint;
		//we will first remove anchor links if they fit removal criteria
		for each(FocalPointPlasticityAnchor* link in linkInvAnchor.getCellLinkList(newCell)) {
			anchorPoint = link->getAnchorPoint();

			double distance = distInvariantCM(xCMNew, yCMNew, zCMNew, anchorPoint[0], anchorPoint[1], anchorPoint[2], fieldDim, boundaryStrategy);
			int maxDistanceLocal;
			if (functionType == BYCELLTYPE || functionType == BYCELLID) {
				maxDistanceLocal = link->getMaxDistance();
			}
			else if (functionType == GLOBAL) {
				maxDistanceLocal = maxDistance;
			}

			if (distance>maxDistanceLocal) {

				deleteAnchor(newCell, link->getAnchorId());
				linkRemoved = true;
				break;
			}
		}

		if (!linkRemoved) {
			for each(FocalPointPlasticityLink* link in linkInv.getCellLinkList(newCell)) {
				nCell = link->getOtherCell(newCell);

				double xCMNeighbor = nCell->xCM / float(nCell->volume);
				double yCMNeighbor = nCell->yCM / float(nCell->volume);
				double zCMNeighbor = nCell->zCM / float(nCell->volume);
				double distance = distInvariantCM(xCMNew, yCMNew, zCMNew, xCMNeighbor, yCMNeighbor, zCMNeighbor, fieldDim, boundaryStrategy);
				int maxDistanceLocal;
				if (functionType == BYCELLTYPE || functionType == BYCELLID) {
					maxDistanceLocal = link->getMaxDistance();
				}
				else if (functionType == GLOBAL) {
					maxDistanceLocal = maxDistance;
				}

				if (distance>maxDistanceLocal) {
					deleteFocalPointPlasticityLink(newCell, nCell);
					break;
				}
			}
		}

		//go over compartments

		for each(FocalPointPlasticityInternalLink* link in linkInvInternal.getCellLinkList(newCell)) {
			nCell = link->getOtherCell(newCell);
			//we remove only one cell at a time even though we could do it for many cells 
			double xCMNeighbor = nCell->xCM / float(nCell->volume);
			double yCMNeighbor = nCell->yCM / float(nCell->volume);
			double zCMNeighbor = nCell->zCM / float(nCell->volume);
			double distance = distInvariantCM(xCMNew, yCMNew, zCMNew, xCMNeighbor, yCMNeighbor, zCMNeighbor, fieldDim, boundaryStrategy);
			int maxDistanceLocal;
			if (functionType == BYCELLTYPE || functionType == BYCELLID) {
				maxDistanceLocal = link->getMaxDistance();
			}
			else if (functionType == GLOBAL) {
				maxDistanceLocal = maxDistance;
			}

			if (distance>maxDistanceLocal) {
				deleteInternalFocalPointPlasticityLink(newCell, nCell);
				break;
			}

		}

	}

	if(oldCell){
		double xCMOld=oldCell->xCM/float(oldCell->volume);
		double yCMOld=oldCell->yCM/float(oldCell->volume);
		double zCMOld=oldCell->zCM/float(oldCell->volume);
		//we remove only one cell at a time even though we could do it for many cells 
		bool linkRemoved = false;
		std::vector<float> anchorPoint;
		//we will first remove anchor links if they fit removal criteria
		for each(FocalPointPlasticityAnchor* link in linkInvAnchor.getCellLinkList(oldCell)) {
			anchorPoint = link->getAnchorPoint();

			double distance = distInvariantCM(xCMOld, yCMOld, zCMOld, anchorPoint[0], anchorPoint[1], anchorPoint[2], fieldDim, boundaryStrategy);
			int maxDistanceLocal;
			if (functionType == BYCELLTYPE || functionType == BYCELLID) {
				maxDistanceLocal = link->getMaxDistance();
			}
			else if (functionType == GLOBAL) {
				maxDistanceLocal = maxDistance;
			}

			if (distance>maxDistanceLocal) {
				deleteAnchor(oldCell, link->getAnchorId());
				linkRemoved = true;
				break;
			}

		}

		if (!linkRemoved) {
			for each(FocalPointPlasticityLink* link in linkInv.getCellLinkList(oldCell)) {
				nCell = link->getOtherCell(oldCell);

				double xCMNeighbor = nCell->xCM / float(nCell->volume);
				double yCMNeighbor = nCell->yCM / float(nCell->volume);
				double zCMNeighbor = nCell->zCM / float(nCell->volume);
				double distance = distInvariantCM(xCMOld, yCMOld, zCMOld, xCMNeighbor, yCMNeighbor, zCMNeighbor, fieldDim, boundaryStrategy);
				//double distance=dist(xCMOld,yCMOld,zCMOld,xCMNeighbor,yCMNeighbor,zCMNeighbor);
				int maxDistanceLocal;
				if (functionType == BYCELLTYPE || functionType == BYCELLID) {
					maxDistanceLocal = link->getMaxDistance();
				}
				else if (functionType == GLOBAL) {
					maxDistanceLocal = maxDistance;
				}

				if (distance>maxDistanceLocal) {
					deleteFocalPointPlasticityLink(oldCell, nCell);
					break;
				}
			}

		}

		//go over compartments
		for each(FocalPointPlasticityInternalLink* link in linkInvInternal.getCellLinkList(oldCell)) {
			nCell = link->getOtherCell(oldCell);
			//we remove only one cell at a time even though we could do it for many cells many cells
			double xCMNeighbor = nCell->xCM / float(nCell->volume);
			double yCMNeighbor = nCell->yCM / float(nCell->volume);
			double zCMNeighbor = nCell->zCM / float(nCell->volume);
			double distance = distInvariantCM(xCMOld, yCMOld, zCMOld, xCMNeighbor, yCMNeighbor, zCMNeighbor, fieldDim, boundaryStrategy);
			//double distance=dist(xCMOld,yCMOld,zCMOld,xCMNeighbor,yCMNeighbor,zCMNeighbor);
			int maxDistanceLocal;
			if (functionType == BYCELLTYPE || functionType == BYCELLID) {
				maxDistanceLocal = link->getMaxDistance();
			}
			else if (functionType == GLOBAL) {
				maxDistanceLocal = maxDistance;
			}

			if (distance>maxDistanceLocal) {
				deleteInternalFocalPointPlasticityLink(oldCell, nCell);
				break;
			}
		}
	}

}

void FocalPointPlasticityPlugin::setFocalPointPlasticityParameters(CellG * _cell1,CellG * _cell2,double _lambda, double _targetDistance,double _maxDistance){

	std::set<FocalPointPlasticityTrackerData> & plastNeighbors1=focalPointPlasticityTrackerAccessor.get(_cell1->extraAttribPtr)->focalPointPlasticityNeighbors;	
	std::set<FocalPointPlasticityTrackerData>::iterator sitr1;
	sitr1=plastNeighbors1.find(FocalPointPlasticityTrackerData(_cell2));
	if(sitr1!=plastNeighbors1.end()){
		//if (sitr1->anchor)
		//	continue;
		//dirty solution to manipulate class stored in a set
		(const_cast<FocalPointPlasticityTrackerData & >(*sitr1)).lambdaDistance=_lambda;
		if(_targetDistance!=0.0){
			(const_cast<FocalPointPlasticityTrackerData & >(*sitr1)).targetDistance=_targetDistance;
		}
		if(_maxDistance!=0.0){
			(const_cast<FocalPointPlasticityTrackerData & >(*sitr1)).maxDistance=_maxDistance;
		}

		//have to change entries in _cell2 for plasticity data associated with _cell1
		std::set<FocalPointPlasticityTrackerData> & plastNeighbors2=focalPointPlasticityTrackerAccessor.get(_cell2->extraAttribPtr)->focalPointPlasticityNeighbors;	
		std::set<FocalPointPlasticityTrackerData>::iterator sitr2;
		sitr2=plastNeighbors2.find(FocalPointPlasticityTrackerData(_cell1));
		if(sitr2!=plastNeighbors2.end()){
			//if (sitr2->anchor)
			//	continue;

			(const_cast<FocalPointPlasticityTrackerData & >(*sitr2)).lambdaDistance=_lambda;
			if(_targetDistance!=0.0){
				(const_cast<FocalPointPlasticityTrackerData & >(*sitr2)).targetDistance=_targetDistance;
			}
			if(_maxDistance!=0.0){
				(const_cast<FocalPointPlasticityTrackerData & >(*sitr2)).maxDistance=_maxDistance;
			}
		}
	}

	FocalPointPlasticityLink* link = linkInv.getLinkByCells(_cell1, _cell2);
	if (link) {
		link->setLambdaDistance(_lambda);
		if (_targetDistance != 0.0) link->setTargetDistance(_targetDistance);
		if (_maxDistance != 0.0) link->setMaxDistance(_maxDistance);
	}
}

void FocalPointPlasticityPlugin::setInternalFocalPointPlasticityParameters(CellG * _cell1,CellG * _cell2,double _lambda, double _targetDistance,double _maxDistance){

	std::set<FocalPointPlasticityTrackerData> & plastNeighbors1=focalPointPlasticityTrackerAccessor.get(_cell1->extraAttribPtr)->internalFocalPointPlasticityNeighbors;	
	std::set<FocalPointPlasticityTrackerData>::iterator sitr1;
	sitr1=plastNeighbors1.find(FocalPointPlasticityTrackerData(_cell2));
	if(sitr1!=plastNeighbors1.end()){
		//if (sitr1->anchor)
		//	continue;
		//dirty solution to manipulate class stored in a set
		(const_cast<FocalPointPlasticityTrackerData & >(*sitr1)).lambdaDistance=_lambda;
		if(_targetDistance!=0.0){
			(const_cast<FocalPointPlasticityTrackerData & >(*sitr1)).targetDistance=_targetDistance;
		}
		if(_maxDistance!=0.0){
			(const_cast<FocalPointPlasticityTrackerData & >(*sitr1)).maxDistance=_maxDistance;
		}

		//have to change entries in _cell2 for plasticity data associated with _cell1
		std::set<FocalPointPlasticityTrackerData> & plastNeighbors2=focalPointPlasticityTrackerAccessor.get(_cell2->extraAttribPtr)->internalFocalPointPlasticityNeighbors;	
		std::set<FocalPointPlasticityTrackerData>::iterator sitr2;
		sitr2=plastNeighbors2.find(FocalPointPlasticityTrackerData(_cell1));
		if(sitr2!=plastNeighbors2.end()){
			//if (sitr2->anchor)
			//	continue;

			(const_cast<FocalPointPlasticityTrackerData & >(*sitr2)).lambdaDistance=_lambda;
			if(_targetDistance!=0.0){
				(const_cast<FocalPointPlasticityTrackerData & >(*sitr2)).targetDistance=_targetDistance;
			}
			if(_maxDistance!=0.0){
				(const_cast<FocalPointPlasticityTrackerData & >(*sitr2)).maxDistance=_maxDistance;
			}
		}
	}

	FocalPointPlasticityInternalLink* link = linkInvInternal.getLinkByCells(_cell1, _cell2);
	if (link) {
		link->setLambdaDistance(_lambda);
		if (_targetDistance != 0.0) link->setTargetDistance(_targetDistance);
		if (_maxDistance != 0.0) link->setMaxDistance(_maxDistance);
	}
}

double FocalPointPlasticityPlugin::getPlasticityParametersLambdaDistance(CellG * _cell1,CellG * _cell2){

	FocalPointPlasticityLink* link = linkInv.getLinkByCells(_cell1, _cell2);
	if (link) {
		return link->getLambdaDistance();
	}
	else {
		return 0.0;
	}
}

double FocalPointPlasticityPlugin::getPlasticityParametersTargetDistance(CellG * _cell1,CellG * _cell2){

	FocalPointPlasticityLink* link = linkInv.getLinkByCells(_cell1, _cell2);
	if (link) {
		return link->getTargetDistance();
	}
	else {
		return 0.0;
	}
}

int FocalPointPlasticityPlugin::createAnchor(CellG * _cell, double _lambda, double _targetDistance,double _maxDistance,float _x, float _y, float _z){
	std::set<FocalPointPlasticityTrackerData> & anchorsSet=focalPointPlasticityTrackerAccessor.get(_cell->extraAttribPtr)->anchors;
	int newAnchorId = linkInvAnchor.getNextAnchorId(_cell);
	
	//cerr<<"newAnchorId="<<newAnchorId<<endl;
	FocalPointPlasticityTrackerData fpptd(0,_lambda, _targetDistance, _maxDistance);
	fpptd.anchor=true;
	fpptd.anchorId=newAnchorId;
	fpptd.anchorPoint[0]=_x;
	fpptd.anchorPoint[1]=_y;
	fpptd.anchorPoint[2]=_z;
	anchorsSet.insert(fpptd);

	FocalPointPlasticityAnchor link = FocalPointPlasticityAnchor(_cell, potts, fpptd);
	linkInvAnchor.addToInventory(&link);

	//cerr<<"anchorsSet.size()="<<anchorsSet.size()<<endl;
	return newAnchorId;

}

void FocalPointPlasticityPlugin::deleteAnchor(CellG * _cell, int _anchorId){
	std::set<FocalPointPlasticityTrackerData> & anchorsSet=focalPointPlasticityTrackerAccessor.get(_cell->extraAttribPtr)->anchors;
	FocalPointPlasticityTrackerData fpptd(0);
	fpptd.anchorId=_anchorId;
	std::set<FocalPointPlasticityTrackerData>::iterator sitr=anchorsSet.find(fpptd);
	if (sitr!=anchorsSet.end()){
		anchorsSet.erase(fpptd);
	}
	FocalPointPlasticityAnchor* link = linkInvAnchor.getAnchor(_cell, _anchorId);
	if (link) linkInvAnchor.removeFromInventory(link);
}

void FocalPointPlasticityPlugin::setAnchorParameters(CellG * _cell, int _anchorId,double _lambda, double _targetDistance,double _maxDistance,float _x, float _y, float _z){
	std::set<FocalPointPlasticityTrackerData> & anchorsSet=focalPointPlasticityTrackerAccessor.get(_cell->extraAttribPtr)->anchors;
	FocalPointPlasticityTrackerData fpptd(0);
	fpptd.anchorId=_anchorId;
	std::set<FocalPointPlasticityTrackerData>::iterator sitr=anchorsSet.find(fpptd);
	if (sitr!=anchorsSet.end()){
		(const_cast<FocalPointPlasticityTrackerData & >(*sitr)).lambdaDistance=_lambda;
		if(_targetDistance!=0.0){
			(const_cast<FocalPointPlasticityTrackerData & >(*sitr)).targetDistance=_targetDistance;
		}
		if(_maxDistance!=0.0){
			(const_cast<FocalPointPlasticityTrackerData & >(*sitr)).maxDistance=_maxDistance;
		}

		if (_x!=-1){
			(const_cast<FocalPointPlasticityTrackerData & >(*sitr)).anchorPoint[0]=_x;
		}

		if (_y!=-1){
			(const_cast<FocalPointPlasticityTrackerData & >(*sitr)).anchorPoint[1]=_y;
		}

		if (_z!=-1){
			(const_cast<FocalPointPlasticityTrackerData & >(*sitr)).anchorPoint[2]=_z;
		}

	}

	FocalPointPlasticityAnchor* link = linkInvAnchor.getAnchor(_cell, _anchorId);
	if (link) {
		link->setLambdaDistance(_lambda);
		if (_targetDistance != 0.0) link->setTargetDistance(_targetDistance);
		if (_maxDistance != 0.0) link->setMaxDistance(_maxDistance);

		std::vector<float> anchorPoint(link->getAnchorPoint());
		if (_x != -1) anchorPoint[0] = _x;
		if (_y != -1) anchorPoint[1] = _y;
		if (_z != -1) anchorPoint[2] = _z;
		link->setAnchorPoint(anchorPoint);
	}

}

std::string FocalPointPlasticityPlugin::steerableName(){return "FocalPointPlasticity";}
std::string FocalPointPlasticityPlugin::toString(){return steerableName();}

int FocalPointPlasticityPlugin::getIndex(const int type1, const int type2) const {
	if (type1 < type2) return ((type1 + 1) | ((type2 + 1) << 16));
	else return ((type2 + 1) | ((type1 + 1) << 16));
}
