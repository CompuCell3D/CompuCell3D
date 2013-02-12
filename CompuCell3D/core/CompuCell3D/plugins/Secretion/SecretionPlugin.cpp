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

// // // #include <CompuCell3D/ClassRegistry.h>

using namespace CompuCell3D;

#include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>
// // // #include <CompuCell3D/Automaton/Automaton.h>
#include <CompuCell3D/plugins/PixelTracker/PixelTrackerPlugin.h>
#include <CompuCell3D/plugins/BoundaryPixelTracker/BoundaryPixelTrackerPlugin.h>
#include <CompuCell3D/steppables/BoxWatcher/BoxWatcher.h>
// // // #include <PublicUtilities/ParallelUtilsOpenMP.h>


// // // #include <BasicUtils/BasicString.h>
// // // #include <BasicUtils/BasicException.h>


using namespace std;


#include "SecretionPlugin.h"
#include "FieldSecretor.h"

SecretionPlugin::SecretionPlugin():
sim(0),
potts(0), 
xmlData(0),
cellFieldG(0),
automaton(0),
boxWatcherSteppable(0),
pUtils(0),
boundaryStrategy(0),
maxNeighborIndex(0),
pixelTrackerPlugin(0),
boundaryPixelTrackerPlugin(0),
disablePixelTracker(false),
disableBoundaryPixelTracker(false)
{}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
SecretionPlugin::~SecretionPlugin() 
{}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SecretionPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

	xmlData=_xmlData;
	sim=simulator;

	potts = sim->getPotts();
	cellFieldG=(WatchableField3D<CellG*> *)potts->getCellFieldG();

	fieldDim=cellFieldG->getDim();

	pUtils=simulator->getParallelUtils();

	automaton=potts->getAutomaton();

	potts->registerFixedStepper(this,true); //by putting true flag as second argument I ensure that secretion fixed stepper will be registered as first module
											//This is quick hack though not a very robust solution. We have to write CC3D scheduler...

	sim->registerSteerableObject(this);

	boundaryStrategy=BoundaryStrategy::getInstance();
	maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);


	bool comPluginAlreadyRegisteredFlag;
	Plugin *plugin=Simulator::pluginManager.get("CenterOfMass",&comPluginAlreadyRegisteredFlag); //this will load Center Of Mass plugin if not already loaded
	if(!comPluginAlreadyRegisteredFlag)
		plugin->init(simulator);
	


}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SecretionPlugin::extraInit(Simulator *simulator) { 

	update(xmlData,true);

	bool useBoxWatcher=false;
	for (int i = 0 ; i < secretionDataPVec.size() ; ++i){
		if(secretionDataPVec[i].useBoxWatcher){
			useBoxWatcher=true;
			break;
		}
	}
	bool steppableAlreadyRegisteredFlag;
	if(useBoxWatcher){
		boxWatcherSteppable=(BoxWatcher*)Simulator::steppableManager.get("BoxWatcher",&steppableAlreadyRegisteredFlag);
		if(!steppableAlreadyRegisteredFlag)
			boxWatcherSteppable->init(simulator);
	}

	bool pixelTrackerAlreadyRegisteredFlag;
	if (! disablePixelTracker){
		pixelTrackerPlugin=(PixelTrackerPlugin*)Simulator::pluginManager.get("PixelTracker",&pixelTrackerAlreadyRegisteredFlag);
		if (!pixelTrackerAlreadyRegisteredFlag){
			pixelTrackerPlugin->init(simulator);
		}
	}

	bool boundaryPixelTrackerAlreadyRegisteredFlag;
	if (! disableBoundaryPixelTracker){
		boundaryPixelTrackerPlugin=(BoundaryPixelTrackerPlugin*)Simulator::pluginManager.get("BoundaryPixelTracker",&boundaryPixelTrackerAlreadyRegisteredFlag);
		if (!boundaryPixelTrackerAlreadyRegisteredFlag){
			boundaryPixelTrackerPlugin->init(simulator);
		}
	}


}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Field3DImpl<float>*  SecretionPlugin::getConcentrationFieldByName(std::string _fieldName)
{
	std::map<std::string,Field3DImpl<float>*> & fieldMap=sim->getConcentrationFieldNameMap();	  
	std::map<std::string,Field3DImpl<float>*>::iterator mitr;
	mitr=fieldMap.find(_fieldName);
	if(mitr!=fieldMap.end()){
		return mitr->second;
	}else{
		return 0;
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldSecretor SecretionPlugin::getFieldSecretor(std::string _fieldName){
	
	FieldSecretor fieldSecretor;

	fieldSecretor.concentrationFieldPtr=getConcentrationFieldByName(_fieldName);
	fieldSecretor.pixelTrackerPlugin=pixelTrackerPlugin;
	fieldSecretor.boundaryPixelTrackerPlugin=boundaryPixelTrackerPlugin;
	fieldSecretor.boundaryStrategy=boundaryStrategy;
	fieldSecretor.maxNeighborIndex=maxNeighborIndex;
	fieldSecretor.cellFieldG=cellFieldG;

	return fieldSecretor;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SecretionPlugin::step(){

	// cerr<<"inside STEP"<<endl;
	unsigned int currentStep;
	unsigned int currentAttempt;
	unsigned int numberOfAttempts;


	currentStep=sim->getStep();
	currentAttempt=potts->getCurrentAttempt();
	numberOfAttempts=potts->getNumberOfAttempts();





	for(unsigned int i = 0 ; i < secretionDataPVec.size() ; ++i ){
		int reminder= (numberOfAttempts % (secretionDataPVec[i].timesPerMCS+1));
		int ratio=(numberOfAttempts / (secretionDataPVec[i].timesPerMCS+1));
		if( ! ((currentAttempt-reminder) % ratio ) && currentAttempt>reminder ){
			for(unsigned int j = 0 ; j <secretionDataPVec[i].secretionFcnPtrVec.size() ; ++j){
				(this->*secretionDataPVec[i].secretionFcnPtrVec[j])(i);
			}

			//          (this->*secrDataVec[i].secretionFcnPtrVec[j])(i);
		}
	}


	//   for(int i=0 ; i <solverDataVec.size() ; ++i){
	//      int reminder= (numberOfAttempts % (solverDataVec[i].extraTimesPerMC+1));
	//	   //cerr<<"reminder="<<reminder<<" numberOfAttampts="<<numberOfAttempts<<" solverDataVec[i].extraTimesPerMC="<<solverDataVec[i].extraTimesPerMC<<" currentAttempt="<<currentAttempt<<endl;   
	//      int ratio=(numberOfAttempts / (solverDataVec[i].extraTimesPerMC+1));
	////       cerr<<"pscpdPtr->solverDataVec[i].extraTimesPerMC="<<pscpdPtr->solverDataVec[i].extraTimesPerMC<<endl;
	//       //cerr<<"ratio="<<ratio<<" reminder="<<reminder<<endl;
	//      if( ! ((currentAttempt-reminder) % ratio ) && currentAttempt>reminder ){
	////          cerr<<"before calling step"<<endl;
	//          solverPtrVec[i]->step(currentStep);
	////          float a=reminder+ratio;
	//        //cerr<<"calling Solver"<<solverDataVec[i].solverName<<" currentAttempt="<<currentAttempt<<" numberOfAttempts="<<numberOfAttempts<<endl;
	//
	//      }
	//
	//   }


}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SecretionPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

	//solverPtrVec.clear();
	//ClassRegistry *classRegistry=sim->getClassRegistry();
	//Steppable * steppable;

	secretionDataPVec.clear();
	if (_xmlData->findElement("DisablePixelTracker")){
		disablePixelTracker=true;
	}
	if (_xmlData->findElement("DisableBoundaryPixelTracker")){
		disableBoundaryPixelTracker=true;
	}

	CC3DXMLElementList secrXMLVec=_xmlData->getElements("Field");
	for(unsigned int i = 0 ; i < secrXMLVec.size() ; ++i ){

		secretionDataPVec.push_back(SecretionDataP());

		SecretionDataP & secrData=secretionDataPVec[secretionDataPVec.size()-1];		
		secrData.update( secrXMLVec[i] );
		secrData.setAutomaton(potts->getAutomaton());

	}

	for(unsigned int i = 0 ; i < secretionDataPVec.size() ; ++i ){
		secretionDataPVec[i].initialize(potts->getAutomaton());
	}


	// CC3DXMLElementList pdeSolversXMLList=_xmlData->getElements("CallPDE");
	// for(unsigned int i=0; i < pdeSolversXMLList.size() ; ++i ){
	// solverDataVec.push_back(SolverData(pdeSolversXMLList[i]->getAttribute("PDESolverName"),pdeSolversXMLList[i]->getAttributeAsUInt("ExtraTimesPerMC")));
	// SolverData & sd=solverDataVec[solverDataVec.size()-1];

	// steppable=classRegistry->getStepper(sd.solverName);
	// solverPtrVec.push_back(steppable);

	// }

}

//bool SecretionPlugin::secreteInsideCell(CellG * _cell, string _fieldName, float _amount){
//	if (!pixelTrackerPlugin){
//		return false;
//	}
//	BasicClassAccessor<PixelTracker> *pixelTrackerAccessorPtr=pixelTrackerPlugin->getPixelTrackerAccessorPtr();
//	set<PixelTrackerData > & pixelSetRef=pixelTrackerAccessorPtr->get(_cell->extraAttribPtr)->pixelSet;
//
//	Field3DImpl<float> & concentrationField=*getConcentrationFieldByName(_fieldName);
//	
//	for (set<PixelTrackerData>::iterator sitr=pixelSetRef.begin() ; sitr!=pixelSetRef.end(); ++sitr){		
//
//		concentrationField.set(sitr->pixel,concentrationField.get(sitr->pixel)+_amount);
//
//	}
//
//	return true;
//}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SecretionPlugin::secreteSingleField(unsigned int idx){

	SecretionDataP & secrData=secretionDataPVec[idx];
	//cerr<<"secrData.typeIdSecrConstMap.size()="<<secrData.typeIdSecrConstMap.size()<<endl;

	float maxUptakeInMedium=0.0;
	float relativeUptakeRateInMedium=0.0;
	float secrConstMedium=0.0;

	std::map<unsigned char,float>::iterator mitrShared;
	std::map<unsigned char,float>::iterator end_mitr=secrData.typeIdSecrConstMap.end();
	std::map<unsigned char,UptakeDataP>::iterator mitrUptakeShared;
	std::map<unsigned char,UptakeDataP>::iterator end_mitrUptake=secrData.typeIdUptakeDataMap.end();



	Field3DImpl<float> & concentrationField=*getConcentrationFieldByName(secrData.fieldName);	

	//cerr<<"concentrationField="<<getConcentrationFieldByName(secrData.fieldName)<<endl;

	bool doUptakeFlag=false;
	bool uptakeInMediumFlag=false;
	bool secreteInMedium=false;
	//the assumption is that medium has type ID 0
	mitrShared=secrData.typeIdSecrConstMap.find(automaton->getTypeId("Medium"));

	if( mitrShared != end_mitr){
		secreteInMedium=true;
		secrConstMedium=mitrShared->second;
	}

	//uptake for medium setup
	if(secrData.typeIdUptakeDataMap.size()){
		doUptakeFlag=true;
	}
	//uptake for medium setup
	if(doUptakeFlag){
		mitrUptakeShared=secrData.typeIdUptakeDataMap.find(automaton->getTypeId("Medium"));
		if(mitrUptakeShared != end_mitrUptake){
			maxUptakeInMedium=mitrUptakeShared->second.maxUptake;
			relativeUptakeRateInMedium=mitrUptakeShared->second.relativeUptakeRate;
			uptakeInMediumFlag=true;

		}
	}



	//	//HAVE TO WATCH OUT FOR SHARED/PRIVATE VARIABLES

	if(secrData.useBoxWatcher){

		unsigned x_min=1,x_max=fieldDim.x+1;
		unsigned y_min=1,y_max=fieldDim.y+1;
		unsigned z_min=1,z_max=fieldDim.z+1;

		Dim3D minDimBW;		
		Dim3D maxDimBW;
		Point3D minCoordinates=*(boxWatcherSteppable->getMinCoordinatesPtr());
		Point3D maxCoordinates=*(boxWatcherSteppable->getMaxCoordinatesPtr());
		//cerr<<"FLEXIBLE DIFF SOLVER maxCoordinates="<<maxCoordinates<<" minCoordinates="<<minCoordinates<<endl;
		x_min=minCoordinates.x+1;
		x_max=maxCoordinates.x+1;
		y_min=minCoordinates.y+1;
		y_max=maxCoordinates.y+1;
		z_min=minCoordinates.z+1;
		z_max=maxCoordinates.z+1;

		minDimBW=Dim3D(x_min,y_min,z_min);
		maxDimBW=Dim3D(x_max,y_max,z_max);
		pUtils->calculateFESolverPartitionWithBoxWatcher(minDimBW,maxDimBW);


	}

	//cerr<<"SECRETE SINGLE FIELD"<<endl;


	pUtils->prepareParallelRegionFESolvers(secrData.useBoxWatcher);


#pragma omp parallel
	{	

		CellG *currentCellPtr;
		//Field3DImpl<float> * concentrationField=concentrationFieldVector[idx];
		float currentConcentration;
		float secrConst;


		std::map<unsigned char,float>::iterator mitr;
		std::map<unsigned char,UptakeDataP>::iterator mitrUptake;

		Point3D pt;
		int threadNumber=pUtils->getCurrentWorkNodeNumber();

		Dim3D minDim;		
		Dim3D maxDim;

		if(secrData.useBoxWatcher){
			minDim=pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).first;
			maxDim=pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).second;

		}else{
			minDim=pUtils->getFESolverPartition(threadNumber).first;
			maxDim=pUtils->getFESolverPartition(threadNumber).second;
		}
		//correcting for (1,1,1) shift in concetration fields used by solvers
		minDim-=Dim3D(1,1,1);
		maxDim-=Dim3D(1,1,1);

		//Dim3D minDim;		
		//Dim3D maxDim=fieldDim;
		//cerr<<"minDim="<<minDim<<" maxDim="<<maxDim<<endl;



		for (int z = minDim.z; z < maxDim.z; z++)
			for (int y = minDim.y; y < maxDim.y; y++)
				for (int x = minDim.x; x < maxDim.x; x++){

					pt=Point3D(x,y,z);
					//cerr<<"pt="<<pt<<" is valid "<<cellFieldG->isValid(pt)<<endl;
					///**
					currentCellPtr=cellFieldG->getQuick(pt);
					//             currentCellPtr=cellFieldG->get(pt);
					//cerr<<"THIS IS PTR="<<currentCellPtr<<endl;

					//             if(currentCellPtr)
					//                cerr<<"This is id="<<currentCellPtr->id<<endl;
					//currentConcentration = concentrationField.getDirect(x,y,z);

					currentConcentration = concentrationField.get(pt);

					if(secreteInMedium && ! currentCellPtr){
						concentrationField.set(pt,currentConcentration+secrConstMedium);
					}

					if(currentCellPtr){											
						mitr=secrData.typeIdSecrConstMap.find(currentCellPtr->type);
						if(mitr!=end_mitr){
							secrConst=mitr->second;
							//cerr<<"secrConst="<<endl;
							//cerr<<"secrData.typeIdSecrConstMap.size()="<<secrData.typeIdSecrConstMap.size()<<endl;
							concentrationField.set(pt,currentConcentration+secrConst);


						}
					}

					if(doUptakeFlag){

						if(uptakeInMediumFlag && ! currentCellPtr){						
							if(currentConcentration*relativeUptakeRateInMedium>maxUptakeInMedium){
								concentrationField.set(pt,concentrationField.get(pt)-maxUptakeInMedium);
							}else{
								concentrationField.set(pt,concentrationField.get(pt) - currentConcentration*relativeUptakeRateInMedium);
							}
						}
						if(currentCellPtr){

							mitrUptake=secrData.typeIdUptakeDataMap.find(currentCellPtr->type);
							if(mitrUptake!=end_mitrUptake){								
								if(currentConcentration*mitrUptake->second.relativeUptakeRate > mitrUptake->second.maxUptake){
									concentrationField.set(pt,concentrationField.get(pt)-mitrUptake->second.maxUptake);
									//cerr<<" uptake concentration="<< currentConcentration<<" relativeUptakeRate="<<mitrUptake->second.relativeUptakeRate<<" subtract="<<mitrUptake->second.maxUptake<<endl;
								}else{
									concentrationField.set(pt,concentrationField.get(pt)-currentConcentration*mitrUptake->second.relativeUptakeRate);
									//cerr<<"concentration="<< currentConconcentrationField.getDirect(x,y,z)- currentConcentration*mitrUptake->second.relativeUptakeRate);
								}
							}
						}
					}
				}
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SecretionPlugin::secreteOnContactSingleField(unsigned int idx){

	SecretionDataP & secrData=secretionDataPVec[idx];

	std::map<unsigned char,SecretionOnContactDataP>::iterator mitrShared;
	std::map<unsigned char,SecretionOnContactDataP>::iterator end_mitr=secrData.typeIdSecrOnContactDataMap.end();



	Field3DImpl<float> & concentrationField=*getConcentrationFieldByName(secrData.fieldName);
	
	std::map<unsigned char, float> * contactCellMapMediumPtr;
	std::map<unsigned char, float> * contactCellMapPtr;


	bool secreteInMedium=false;
	//the assumption is that medium has type ID 0
	mitrShared=secrData.typeIdSecrOnContactDataMap.find(automaton->getTypeId("Medium"));

	if(mitrShared != end_mitr ){
		secreteInMedium=true;
		contactCellMapMediumPtr = &(mitrShared->second.contactCellMap);
	}


	//HAVE TO WATCH OUT FOR SHARED/PRIVATE VARIABLES
	
	
	if(secrData.useBoxWatcher){

		unsigned x_min=1,x_max=fieldDim.x+1;
		unsigned y_min=1,y_max=fieldDim.y+1;
		unsigned z_min=1,z_max=fieldDim.z+1;

		Dim3D minDimBW;		
		Dim3D maxDimBW;
		Point3D minCoordinates=*(boxWatcherSteppable->getMinCoordinatesPtr());
		Point3D maxCoordinates=*(boxWatcherSteppable->getMaxCoordinatesPtr());
		//cerr<<"FLEXIBLE DIFF SOLVER maxCoordinates="<<maxCoordinates<<" minCoordinates="<<minCoordinates<<endl;
		x_min=minCoordinates.x+1;
		x_max=maxCoordinates.x+1;
		y_min=minCoordinates.y+1;
		y_max=maxCoordinates.y+1;
		z_min=minCoordinates.z+1;
		z_max=maxCoordinates.z+1;

		minDimBW=Dim3D(x_min,y_min,z_min);
		maxDimBW=Dim3D(x_max,y_max,z_max);
		pUtils->calculateFESolverPartitionWithBoxWatcher(minDimBW,maxDimBW);

	}

	

pUtils->prepareParallelRegionFESolvers(secrData.useBoxWatcher);
#pragma omp parallel
	{	
		
		std::map<unsigned char,SecretionOnContactDataP>::iterator mitr;
		std::map<unsigned char, float>::iterator mitrTypeConst;
		
		float currentConcentration;
		float secrConst;
		float secrConstMedium=0.0;

		CellG *currentCellPtr;
		Point3D pt;
		Neighbor n;
		CellG *nCell=0;
		WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
		unsigned char type;

		int threadNumber=pUtils->getCurrentWorkNodeNumber();

		Dim3D minDim;		
		Dim3D maxDim;

		if(secrData.useBoxWatcher){
			minDim=pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).first;
			maxDim=pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).second;

		}else{
			minDim=pUtils->getFESolverPartition(threadNumber).first;
			maxDim=pUtils->getFESolverPartition(threadNumber).second;
		}
		//correcting for (1,1,1) shift in concetration fields used by solvers
		minDim-=Dim3D(1,1,1);
		maxDim-=Dim3D(1,1,1);



		for (int z = minDim.z; z < maxDim.z; z++)
			for (int y = minDim.y; y < maxDim.y; y++)
				for (int x = minDim.x; x < maxDim.x; x++){
				pt=Point3D(x,y,z);
				///**
				currentCellPtr=cellFieldG->getQuick(pt);
				//             currentCellPtr=cellFieldG->get(pt);
				currentConcentration = concentrationField.get(pt);

				if(secreteInMedium && ! currentCellPtr){
					for (int i = 0  ; i<=maxNeighborIndex/*offsetVec.size()*/ ; ++i ){
						n=boundaryStrategy->getNeighborDirect(pt,i);
						if(!n.distance)//not a valid neighbor
							continue;
						///**
						nCell = fieldG->get(n.pt);
						//                      nCell = fieldG->get(n.pt);
						if(nCell)
							type=nCell->type;
						else
							type=0;

						mitrTypeConst=contactCellMapMediumPtr->find(type);

						if(mitrTypeConst != contactCellMapMediumPtr->end()){//OK to secrete, contact detected
							secrConstMedium = mitrTypeConst->second;

							concentrationField.set(pt,currentConcentration+secrConstMedium);
						}
					}
					continue;
				}

				if(currentCellPtr){
					mitr=secrData.typeIdSecrOnContactDataMap.find(currentCellPtr->type);
					if(mitr!=end_mitr){

						contactCellMapPtr = &(mitr->second.contactCellMap);

						for (int i = 0  ; i<=maxNeighborIndex/*offsetVec.size() */; ++i ){

							n=boundaryStrategy->getNeighborDirect(pt,i);
							if(!n.distance)//not a valid neighbor
								continue;
							///**
							nCell = fieldG->get(n.pt);
							//                      nCell = fieldG->get(n.pt);
							if(nCell)
								type=nCell->type;
							else
								type=0;

							if (currentCellPtr==nCell) continue; //skip secretion in pixels belongin to the same cell

							mitrTypeConst=contactCellMapPtr->find(type);
							if(mitrTypeConst != contactCellMapPtr->end()){//OK to secrete, contact detected
								secrConst=mitrTypeConst->second;
								//                         concentrationField->set(pt,currentConcentration+secrConst);
								concentrationField.set(pt,currentConcentration+secrConst);
							}
						}
					}
				}
			}
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SecretionPlugin::secreteConstantConcentrationSingleField(unsigned int idx){

	SecretionDataP & secrData=secretionDataPVec[idx];

	std::map<unsigned char,float>::iterator mitrShared;
	std::map<unsigned char,float>::iterator end_mitr=secrData.typeIdSecrConstConstantConcentrationMap.end();


	float secrConstMedium=0.0;

	Field3DImpl<float> & concentrationField=*getConcentrationFieldByName(secrData.fieldName);
	

	bool secreteInMedium=false;
	//the assumption is that medium has type ID 0
	mitrShared=secrData.typeIdSecrConstConstantConcentrationMap.find(automaton->getTypeId("Medium"));

	if( mitrShared != end_mitr){
		secreteInMedium=true;
		secrConstMedium=mitrShared->second;
	}


	//HAVE TO WATCH OUT FOR SHARED/PRIVATE VARIABLES
	
	if(secrData.useBoxWatcher){

		unsigned x_min=1,x_max=fieldDim.x+1;
		unsigned y_min=1,y_max=fieldDim.y+1;
		unsigned z_min=1,z_max=fieldDim.z+1;

		Dim3D minDimBW;		
		Dim3D maxDimBW;
		Point3D minCoordinates=*(boxWatcherSteppable->getMinCoordinatesPtr());
		Point3D maxCoordinates=*(boxWatcherSteppable->getMaxCoordinatesPtr());
		//cerr<<"FLEXIBLE DIFF SOLVER maxCoordinates="<<maxCoordinates<<" minCoordinates="<<minCoordinates<<endl;
		x_min=minCoordinates.x+1;
		x_max=maxCoordinates.x+1;
		y_min=minCoordinates.y+1;
		y_max=maxCoordinates.y+1;
		z_min=minCoordinates.z+1;
		z_max=maxCoordinates.z+1;

		minDimBW=Dim3D(x_min,y_min,z_min);
		maxDimBW=Dim3D(x_max,y_max,z_max);
		pUtils->calculateFESolverPartitionWithBoxWatcher(minDimBW,maxDimBW);

	}

	pUtils->prepareParallelRegionFESolvers(secrData.useBoxWatcher);

#pragma omp parallel
	{	

		CellG *currentCellPtr;
		//Field3DImpl<float> * concentrationField=concentrationFieldVector[idx];
		float currentConcentration;
		float secrConst;
		
		std::map<unsigned char,float>::iterator mitr;

		Point3D pt;
		int threadNumber=pUtils->getCurrentWorkNodeNumber();

		Dim3D minDim;		
		Dim3D maxDim;

		if(secrData.useBoxWatcher){
			minDim=pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).first;
			maxDim=pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).second;

		}else{
			minDim=pUtils->getFESolverPartition(threadNumber).first;
			maxDim=pUtils->getFESolverPartition(threadNumber).second;
		}

		//correcting for (1,1,1) shift in concetration fields used by solvers
		minDim-=Dim3D(1,1,1);
		maxDim-=Dim3D(1,1,1);

		for (int z = minDim.z; z < maxDim.z; z++)
			for (int y = minDim.y; y < maxDim.y; y++)
				for (int x = minDim.x; x < maxDim.x; x++){

				pt=Point3D(x,y,z);
				//             cerr<<"pt="<<pt<<" is valid "<<cellFieldG->isValid(pt)<<endl;
				///**
				currentCellPtr=cellFieldG->getQuick(pt);
				//             currentCellPtr=cellFieldG->get(pt);
				//             cerr<<"THIS IS PTR="<<currentCellPtr<<endl;

				//             if(currentCellPtr)
				//                cerr<<"This is id="<<currentCellPtr->id<<endl;
				//currentConcentration = concentrationArray[x][y][z];

				if(secreteInMedium && ! currentCellPtr){
					concentrationField.set(pt,secrConstMedium);
				}

				if(currentCellPtr){
					mitr=secrData.typeIdSecrConstConstantConcentrationMap.find(currentCellPtr->type);
					if(mitr!=end_mitr){
						secrConst=mitr->second;
						concentrationField.set(pt,secrConst);
					}
				}
			}
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


std::string SecretionPlugin::toString(){

	return "Secretion";

}

std::string SecretionPlugin::steerableName(){

	return toString();

}
