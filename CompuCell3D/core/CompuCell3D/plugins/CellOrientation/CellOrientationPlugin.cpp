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
// // // #include <CompuCell3D/Field3D/Point3D.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
// // // #include <PublicUtilities/NumericalUtils.h>
// // // #include <Utils/Coordinates3D.h>
// // // #include <PublicUtilities/StringUtils.h>
// // // #include <CompuCell3D/Simulator.h>
#include <CompuCell3D/plugins/PolarizationVector/PolarizationVector.h>
#include <CompuCell3D/plugins/PolarizationVector/PolarizationVectorPlugin.h>

using namespace CompuCell3D;


// // // #include <iostream>
using namespace std;


#include "CellOrientationPlugin.h"




CellOrientationPlugin::CellOrientationPlugin() :  potts(0),
		simulator(0),
		cellFieldG(0),
		polarizationVectorAccessorPtr(0),
		lambdaCellOrientation(0.0),
		changeEnergyFcnPtr(&CellOrientationPlugin::changeEnergyPixelBased),
		boundaryStrategy(0),
		lambdaFlexFlag(false)
{
}

void CellOrientationPlugin::setLambdaCellOrientation(CellG * _cell, double _lambda){
	lambdaCellOrientationAccessor.get(_cell->extraAttribPtr)->lambdaVal=_lambda;
}
double CellOrientationPlugin::getLambdaCellOrientation(CellG * _cell){
	return lambdaCellOrientationAccessor.get(_cell->extraAttribPtr)->lambdaVal;
}


CellOrientationPlugin::~CellOrientationPlugin() {

}

void CellOrientationPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

	cerr<<"INITIALIZE CELL ORIENTATION PLUGIN"<<endl;   
	potts = simulator->getPotts();
	//    potts->getCellFactoryGroupPtr()->registerClass(&CellOrientationVectorAccessor); //register new class with the factory

	bool pluginAlreadyRegisteredFlag;
	PolarizationVectorPlugin * polVectorPlugin = (PolarizationVectorPlugin*) Simulator::pluginManager.get("PolarizationVector",&pluginAlreadyRegisteredFlag);
	if(!pluginAlreadyRegisteredFlag)
		polVectorPlugin->init(simulator);

	bool comPluginAlreadyRegisteredFlag;
	Plugin *comPlugin=Simulator::pluginManager.get("CenterOfMass",&comPluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded
	if(!comPluginAlreadyRegisteredFlag)
		comPlugin->init(simulator);

	polarizationVectorAccessorPtr=polVectorPlugin->getPolarizationVectorAccessorPtr();

	cellFieldG = potts->getCellFieldG();

	fieldDim=cellFieldG->getDim();

	boundaryStrategy=BoundaryStrategy::getInstance();

	potts->registerEnergyFunctionWithName(this,"CellOrientationEnergy");

	potts->getCellFactoryGroupPtr()->registerClass(&lambdaCellOrientationAccessor);


	simulator->registerSteerableObject(this);
	update(_xmlData,true);


}

void CellOrientationPlugin::extraInit(Simulator *simulator) {
	cerr<<"EXTRA INITIALIZE CELL ORIENTATION PLUGIN"<<endl;   
	Potts3D *potts = simulator->getPotts();
	cellFieldG = potts->getCellFieldG();
}




void CellOrientationPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

	if (!_xmlData->getNumberOfChildren()){ //using cell id - based lambdaCellOrientation
		lambdaFlexFlag=true;
		return;
	}

	if(_xmlData->findElement("LambdaCellOrientation")){
		lambdaCellOrientation=_xmlData->getFirstElement("LambdaCellOrientation")->getDouble();
	}

	if(_xmlData->findElement("LambdaFlex"))
		lambdaFlexFlag=true;
	else
		lambdaFlexFlag=false;

	bool comBasedAlgorithm=false;
	if(_xmlData->findElement("Algorithm")){ 

		string algorithm=_xmlData->getFirstElement("Algorithm")->getText();

		changeToLower(algorithm);

		if(algorithm=="centerofmassbased"){
			comBasedAlgorithm=true;
			changeEnergyFcnPtr = &CellOrientationPlugin::changeEnergyCOMBased;
		}
	}

}

double CellOrientationPlugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {
	return 0.0;
// 	(this->*changeEnergyFcnPtr)(pt,newCell,oldCell);
}

double CellOrientationPlugin::changeEnergyPixelBased(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {

	float energy=0.0;
	PolarizationVector * polarizationVecPtr;
	Point3D spinCopyVector;

	//spin change takes place at pt  and spin from potts->getFlipNeighbor() is copied to pt. We define spinCopyVector as:
	//    spinCopyVector=pt-potts->getFlipNeighbor();

	//this will return distance vector which will properly account for different boundary conditions   
	spinCopyVector=distanceVectorInvariant(pt,potts->getFlipNeighbor(),fieldDim);


	double lambdaCellOrientationValue=0.0;

	if(oldCell){

		if(!lambdaFlexFlag){
			lambdaCellOrientationValue=lambdaCellOrientation;
		}else{
			lambdaCellOrientationValue=lambdaCellOrientationAccessor.get(oldCell->extraAttribPtr)->lambdaVal;
		}

		polarizationVecPtr = polarizationVectorAccessorPtr->get(oldCell->extraAttribPtr);
		energy+=-lambdaCellOrientationValue*(polarizationVecPtr->x * spinCopyVector.x + polarizationVecPtr->y * spinCopyVector.y + polarizationVecPtr->z * spinCopyVector.z);

	}


	if(newCell){

		if(!lambdaFlexFlag){
			lambdaCellOrientationValue=lambdaCellOrientation;
		}else{
			lambdaCellOrientationValue=lambdaCellOrientationAccessor.get(newCell->extraAttribPtr)->lambdaVal;
		}

		polarizationVecPtr = polarizationVectorAccessorPtr->get(newCell->extraAttribPtr);	  
		energy+=-lambdaCellOrientationValue*(polarizationVecPtr->x * spinCopyVector.x + polarizationVecPtr->y * spinCopyVector.y + polarizationVecPtr->z * spinCopyVector.z);

	}

	//    cerr<<"energy="<<energy<<endl;
	
	return energy;
}


double CellOrientationPlugin::changeEnergyCOMBased(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {

	double energy=0.0;	
	PolarizationVector * polarizationVecPtr;
	double lambdaCellOrientationValue=0.0;


	if (oldCell){
		Coordinates3D<double> oldCOMAfterFlip=precalculateCentroid(pt, oldCell, -1,fieldDim, boundaryStrategy);

		if(oldCell->volume>1){
			oldCOMAfterFlip.XRef()=oldCOMAfterFlip.X()/(float)(oldCell->volume-1);
			oldCOMAfterFlip.YRef()=oldCOMAfterFlip.Y()/(float)(oldCell->volume-1);
			oldCOMAfterFlip.ZRef()=oldCOMAfterFlip.Z()/(float)(oldCell->volume-1);
		}else{

			oldCOMAfterFlip=Coordinates3D<double>(oldCell->xCM/oldCell->volume,oldCell->zCM/oldCell->volume,oldCell->zCM/oldCell->volume);

		}

		if(!lambdaFlexFlag){
			lambdaCellOrientationValue=lambdaCellOrientation;
		}else{
			lambdaCellOrientationValue=lambdaCellOrientationAccessor.get(oldCell->extraAttribPtr)->lambdaVal;
		}

		polarizationVecPtr = polarizationVectorAccessorPtr->get(oldCell->extraAttribPtr);
		

		Coordinates3D<double> oldCOMBeforeFlip(oldCell->xCM/oldCell->volume, oldCell->yCM/oldCell->volume, oldCell->zCM/oldCell->volume);		
		Coordinates3D<double> distVector = distanceVectorCoordinatesInvariant(oldCOMAfterFlip ,oldCOMBeforeFlip,fieldDim);


		//cerr<<"lambdaCellOrientationValue="<<lambdaCellOrientationValue<<endl;		
		//cerr<<"distVector="<<distVector<<endl;
		//cerr<<"p.x="<<polarizationVecPtr->x<<" p.y="<<polarizationVecPtr->y<<" p.z="<<polarizationVecPtr->z<<endl;

		energy += -lambdaCellOrientationValue*(polarizationVecPtr->x * distVector.x + polarizationVecPtr->y * distVector.y + polarizationVecPtr->z * distVector.z);
	}

	
	if (newCell){

		Coordinates3D<double> newCOMAfterFlip=precalculateCentroid(pt, newCell, 1,fieldDim, boundaryStrategy);


		newCOMAfterFlip.XRef()=newCOMAfterFlip.X()/(float)(newCell->volume+1);
		newCOMAfterFlip.YRef()=newCOMAfterFlip.Y()/(float)(newCell->volume+1);
		newCOMAfterFlip.ZRef()=newCOMAfterFlip.Z()/(float)(newCell->volume+1);

		if(!lambdaFlexFlag){
			lambdaCellOrientationValue=lambdaCellOrientation;
		}else{
			lambdaCellOrientationValue=lambdaCellOrientationAccessor.get(newCell->extraAttribPtr)->lambdaVal;
		}

		polarizationVecPtr = polarizationVectorAccessorPtr->get(newCell->extraAttribPtr);	

		Coordinates3D<double> newCOMBeforeFlip(newCell->xCM/newCell->volume, newCell->yCM/newCell->volume, newCell->zCM/newCell->volume);
		Coordinates3D<double> distVector = distanceVectorCoordinatesInvariant(newCOMAfterFlip ,newCOMBeforeFlip,fieldDim);

		energy += -lambdaCellOrientationValue*(polarizationVecPtr->x * distVector.x + polarizationVecPtr->y * distVector.y + polarizationVecPtr->z * distVector.z);

	}
	
	return energy;
}



std::string CellOrientationPlugin::toString(){
	return "CellOrientation";
}


std::string CellOrientationPlugin::steerableName(){
	return toString();
}

