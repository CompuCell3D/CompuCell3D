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

// // // #include <CompuCell3D/Automaton/Automaton.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
// // // #include <CompuCell3D/Field3D/Field3D.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>
// // // #include <CompuCell3D/Boundary/BoundaryStrategy.h>
// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
// // // #include <PublicUtilities/NumericalUtils.h>
// // // #include <complex>
// // // #include <algorithm>

using namespace CompuCell3D;



// // // #include <iostream>
using namespace std;


#include "LengthConstraintPlugin.h"

LengthConstraintPlugin::LengthConstraintPlugin() : xmlData(0),potts(0),changeEnergyFcnPtr(0) 
{}

LengthConstraintPlugin::~LengthConstraintPlugin() {}

void LengthConstraintPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

	xmlData=_xmlData;
	this->simulator=simulator; 
	potts = simulator->getPotts();  

	bool pluginAlreadyRegisteredFlag;
	Plugin *plugin=Simulator::pluginManager.get("MomentOfInertia",&pluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded
	if(!pluginAlreadyRegisteredFlag)
		plugin->init(simulator);

	boundaryStrategy=BoundaryStrategy::getInstance();

	potts->getCellFactoryGroupPtr()->registerClass(&lengthConstraintDataAccessor);
	potts->registerEnergyFunctionWithName(this,"LengthConstraint");


	simulator->registerSteerableObject(this);

	Dim3D fieldDim=potts->getCellFieldG()->getDim();
	if(fieldDim.x==1){
		changeEnergyFcnPtr=&LengthConstraintPlugin::changeEnergy_yz;

	}else if(fieldDim.y==1){
		changeEnergyFcnPtr=&LengthConstraintPlugin::changeEnergy_xz;

	}else if (fieldDim.z==1){
		changeEnergyFcnPtr=&LengthConstraintPlugin::changeEnergy_xy;

	}else{
		changeEnergyFcnPtr=&LengthConstraintPlugin::changeEnergy_3D;

		//ASSERT_OR_THROW("Currently LengthConstraint plugin can only be used in 2D",0);
	}

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void LengthConstraintPlugin::setLengthConstraintData(CellG * _cell, double _lambdaLength, double _targetLength ,double _minorTargetLength){
	if(_cell){
		lengthConstraintDataAccessor.get(_cell->extraAttribPtr)->lambdaLength=_lambdaLength;
		lengthConstraintDataAccessor.get(_cell->extraAttribPtr)->targetLength=_targetLength;
		lengthConstraintDataAccessor.get(_cell->extraAttribPtr)->minorTargetLength=_minorTargetLength;
	}	
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double LengthConstraintPlugin::getLambdaLength(CellG * _cell){
	if (_cell){
		return lengthConstraintDataAccessor.get(_cell->extraAttribPtr)->lambdaLength;
	}
	return 0.0;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double LengthConstraintPlugin::getTargetLength(CellG * _cell){
	if (_cell){
		return lengthConstraintDataAccessor.get(_cell->extraAttribPtr)->targetLength;
	}
	return 0.0;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double LengthConstraintPlugin::getMinorTargetLength(CellG * _cell){
	if (_cell){
		return lengthConstraintDataAccessor.get(_cell->extraAttribPtr)->minorTargetLength;
	}
	return 0.0;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void LengthConstraintPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

	if(potts->getDisplayUnitsFlag()){		
		Unit lambdaLengthUnit=potts->getEnergyUnit()/(potts->getLengthUnit()*potts->getLengthUnit());

		CC3DXMLElement * unitsElem=_xmlData->getFirstElement("Units"); 
		if (!unitsElem){ //add Units element
			unitsElem=_xmlData->attachElement("Units");
		}

		if(unitsElem->getFirstElement("TargetLengthUnit")){
			unitsElem->getFirstElement("TargetLengthUnit")->updateElementValue(potts->getLengthUnit().toString());
		}else{
			unitsElem->attachElement("TargetLengthUnit",potts->getLengthUnit().toString());
		}



		if(unitsElem->getFirstElement("MinorTargetLengthUnit")){
			unitsElem->getFirstElement("MinorTargetLengthUnit")->updateElementValue(potts->getLengthUnit().toString());
		}else{
			unitsElem->attachElement("MinorTargetLengthUnit",potts->getLengthUnit().toString());
		}

		if(unitsElem->getFirstElement("LambdaLengthUnit")){
			unitsElem->getFirstElement("LambdaLengthUnit")->updateElementValue(lambdaLengthUnit.toString());
		}else{
			unitsElem->attachElement("LambdaLengthUnit",lambdaLengthUnit.toString());
		}

	}


	typeNameVec.clear();
	lengthEnergyParamVector.clear();

	CC3DXMLElementList lengthEnergyParamVecXML=_xmlData->getElements("LengthEnergyParameters");
	for (int i =0 ; i < lengthEnergyParamVecXML.size() ; ++i){
		LengthEnergyParam lengthEnergyParam(
			lengthEnergyParamVecXML[i]->getAttribute("CellType"),
			lengthEnergyParamVecXML[i]->getAttributeAsDouble("TargetLength"),
			lengthEnergyParamVecXML[i]->getAttributeAsDouble("LambdaLength")
			);

		if(lengthEnergyParamVecXML[i]->findAttribute("MinorTargetLength") ){
			lengthEnergyParam.minorTargetLength=lengthEnergyParamVecXML[i]->getAttributeAsDouble("MinorTargetLength");
		}

		typeNameVec.push_back(lengthEnergyParam.cellTypeName);
		lengthEnergyParamVector.push_back(lengthEnergyParam);
	}
	//have to make sure that potts ptr is initilized
	ASSERT_OR_THROW("Potts pointer is unitialized",potts);
	initTypeId(potts);
}


void LengthConstraintPlugin::extraInit(Simulator *simulator){
	update(xmlData,true);
}

double LengthConstraintPlugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {


	/// E = lambda * (length - targetLength) ^ 2 

	return (this->*changeEnergyFcnPtr)(pt,newCell,oldCell);


}


double LengthConstraintPlugin::changeEnergy_xz(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {


	// Assumption: COM and Volume has not been updated.

	/// E = lambda * (length - targetLength) ^ 2 

	//Center of mass, length constraints calculations are done withou checking whether cell volume reaches 0 or not
	// when cell is about to disappear this results in Nan values of energy - because division by 0 is involved or
	// sqrt(expression involving compoinents of inertia tensor) is NaN
	//in all such cases we set energy to 0 i.e. if energy=Nan we set it to energy=0.0

	double energy = 0.0;

	if (oldCell == newCell) return 0.0;

	Coordinates3D<double> ptTrans=boundaryStrategy->calculatePointCoordinates(pt);  
	//as in the original version 
	if (newCell){

		//local definitions of length constraint have priority over by type definitions
		double lambdaLength=lengthConstraintDataAccessor.get(newCell->extraAttribPtr)->lambdaLength;;
		double targetLength=lengthConstraintDataAccessor.get(newCell->extraAttribPtr)->targetLength;;
		
		if(lambdaLength==0.0) {
			if ( newCell->type < lengthEnergyParamVector.size() ){
				lambdaLength=lengthEnergyParamVector[newCell->type].lambdaLength;
				targetLength=lengthEnergyParamVector[newCell->type].targetLength;
			}
		}
		//we can optimize it further in case user does not specify local paramteress (i.e. per cell id and by-type definition is not specified as well)

		double xcm = (newCell->xCM / (float) newCell->volume);
		double zcm = (newCell->zCM / (float) newCell->volume);
		double newXCM = (newCell->xCM + ptTrans.x)/((float)newCell->volume + 1);
		double newZCM = (newCell->zCM + ptTrans.z)/((float)newCell->volume + 1);

		double newIxx=newCell->iXX+(newCell->volume )*zcm*zcm-(newCell->volume+1)*(newZCM*newZCM)+ptTrans.z*ptTrans.z;
		double newIzz=newCell->iZZ+(newCell->volume )*xcm*xcm-(newCell->volume+1)*(newXCM*newXCM)+ptTrans.x*ptTrans.x;
		double newIxz=newCell->iXZ-(newCell->volume )*xcm*zcm+(newCell->volume+1)*newXCM*newZCM-ptTrans.x*ptTrans.z;

		double currLength = 4.0*sqrt(((float)((0.5*(newCell->iXX + newCell->iZZ)) + .5*sqrt((float)((newCell->iXX - newCell->iZZ)*(newCell->iXX - newCell->iZZ) + 4*(newCell->iXZ)*(newCell->iXZ)))))/(float)(newCell->volume));

		double currEnergy = lambdaLength * (currLength - targetLength)*(currLength - targetLength);
		double newLength = 4.0*sqrt(((float)((0.5*(newIxx + newIzz)) + .5*sqrt((float)((newIxx - newIzz)*(newIxx - newIzz) + 4*newIxz*newIxz))))/(float)(newCell->volume+1));
		double newEnergy = lambdaLength * (newLength - targetLength)*(newLength - targetLength);
		energy += newEnergy - currEnergy;
	}
	if (oldCell) {
		//local definitions of length constraint have priority over by type definitions
		double lambdaLength=lengthConstraintDataAccessor.get(oldCell->extraAttribPtr)->lambdaLength;;
		double targetLength=lengthConstraintDataAccessor.get(oldCell->extraAttribPtr)->targetLength;;
		
		if(lambdaLength==0.0) {
			if ( oldCell->type < lengthEnergyParamVector.size() ){
				lambdaLength=lengthEnergyParamVector[oldCell->type].lambdaLength;
				targetLength=lengthEnergyParamVector[oldCell->type].targetLength;
			}
		}

		double xcm = (oldCell->xCM / (float) oldCell->volume);
		double zcm = (oldCell->zCM / (float) oldCell->volume);
		double newXCM = (oldCell->xCM - ptTrans.x)/((float)oldCell->volume - 1);
		double newZCM = (oldCell->zCM - ptTrans.z)/((float)oldCell->volume - 1);

		double newIxx =oldCell->iXX+(oldCell->volume )*(zcm*zcm)-(oldCell->volume-1)*(newZCM*newZCM)-ptTrans.z*ptTrans.z;
		double newIzz =oldCell->iZZ+(oldCell->volume )*(xcm*xcm)-(oldCell->volume-1)*(newXCM*newXCM)-ptTrans.x*ptTrans.x;
		double newIxz =oldCell->iXZ-(oldCell->volume )*(xcm*zcm)+(oldCell->volume-1)*newXCM*newZCM+ptTrans.x*ptTrans.z;

		double currLength = 4.0*sqrt(((float)((0.5*(oldCell->iXX + oldCell->iZZ)) + .5*sqrt((float)((oldCell->iXX - oldCell->iZZ)*(oldCell->iXX - oldCell->iZZ) + 4*(oldCell->iXZ)*(oldCell->iXZ)))))/(float)(oldCell->volume));
		double currEnergy = lambdaLength * (currLength - targetLength)*(currLength - targetLength);
		double newLength;
		if(oldCell->volume<=1){
			newLength = 0.0;
		}else{
			newLength = 4.0*sqrt(((float)((0.5*(newIxx + newIzz)) + .5*sqrt((float)((newIxx - newIzz)*(newIxx - newIzz) + 4*newIxz*newIxz))))/(float)(oldCell->volume-1));
		}

		double newEnergy = lambdaLength * (newLength - targetLength) * (newLength - targetLength);
		energy += newEnergy - currEnergy;
	}


	if(energy!=energy)
		return 0.0;
	else
		return energy;
}


double LengthConstraintPlugin::changeEnergy_xy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {


	// Assumption: COM and Volume has not been updated.

	/// E = lambda * (length - targetLength) ^ 2 

	//Center of mass, length constraints calculations are done withou checking whether cell volume reaches 0 or not
	// when cell is about to disappear this results in Nan values of energy - because division by 0 is involved or
	// sqrt(expression involving compoinents of inertia tensor) is NaN
	//in all such cases we set energy to 0 i.e. if energy=Nan we set it to energy=0.0

	double energy = 0.0;

	if (oldCell == newCell) return 0.0;

	Coordinates3D<double> ptTrans=boundaryStrategy->calculatePointCoordinates(pt);  

	//as in the original version 
	if (newCell){
		//local definitions of length constraint have priority over by type definitions
		double lambdaLength=lengthConstraintDataAccessor.get(newCell->extraAttribPtr)->lambdaLength;;
		double targetLength=lengthConstraintDataAccessor.get(newCell->extraAttribPtr)->targetLength;;
		
		if(lambdaLength==0.0) {
			if ( newCell->type < lengthEnergyParamVector.size() ){
				lambdaLength=lengthEnergyParamVector[newCell->type].lambdaLength;
				targetLength=lengthEnergyParamVector[newCell->type].targetLength;
			}
		}

		double xcm = (newCell->xCM / (float) newCell->volume);
		double ycm = (newCell->yCM / (float) newCell->volume);
		double newXCM = (newCell->xCM + ptTrans.x)/((float)newCell->volume + 1);
		double newYCM = (newCell->yCM + ptTrans.y)/((float)newCell->volume + 1);	 

		double newIxx=newCell->iXX+(newCell->volume )*ycm*ycm-(newCell->volume+1)*(newYCM*newYCM)+ptTrans.y*ptTrans.y;
		double newIyy=newCell->iYY+(newCell->volume )*xcm*xcm-(newCell->volume+1)*(newXCM*newXCM)+ptTrans.x*ptTrans.x;
		double newIxy=newCell->iXY-(newCell->volume )*xcm*ycm+(newCell->volume+1)*newXCM*newYCM-ptTrans.x*ptTrans.y;

		double currLength = 4.0*sqrt(((float)((0.5*(newCell->iXX + newCell->iYY)) + .5*sqrt((float)((newCell->iXX - newCell->iYY)*(newCell->iXX - newCell->iYY) + 4*(newCell->iXY)*(newCell->iXY)))))/(float)(newCell->volume));

		double currEnergy = lambdaLength * (currLength - targetLength)*(currLength - targetLength);
		double newLength = 4.0*sqrt(((float)((0.5*(newIxx + newIyy)) + .5*sqrt((float)((newIxx - newIyy)*(newIxx - newIyy) + 4*newIxy*newIxy))))/(float)(newCell->volume+1));
		double newEnergy = lambdaLength * (newLength - targetLength)*(newLength - targetLength);
		energy += newEnergy - currEnergy;
	}
	if (oldCell) {
		//local definitions of length constraint have priority over by type definitions
		double lambdaLength=lengthConstraintDataAccessor.get(oldCell->extraAttribPtr)->lambdaLength;;
		double targetLength=lengthConstraintDataAccessor.get(oldCell->extraAttribPtr)->targetLength;;
		
		if(lambdaLength==0.0) {
			if ( oldCell->type < lengthEnergyParamVector.size() ){
				lambdaLength=lengthEnergyParamVector[oldCell->type].lambdaLength;
				targetLength=lengthEnergyParamVector[oldCell->type].targetLength;
			}
		}

		double xcm = (oldCell->xCM / (float) oldCell->volume);
		double ycm = (oldCell->yCM / (float) oldCell->volume);
		double newXCM = (oldCell->xCM - ptTrans.x)/((float)oldCell->volume - 1);
		double newYCM = (oldCell->yCM - ptTrans.y)/((float)oldCell->volume - 1);

		double newIxx =oldCell->iXX+(oldCell->volume )*(ycm*ycm)-(oldCell->volume-1)*(newYCM*newYCM)-ptTrans.y*ptTrans.y;
		double newIyy =oldCell->iYY+(oldCell->volume )*(xcm*xcm)-(oldCell->volume-1)*(newXCM*newXCM)-ptTrans.x*ptTrans.x;
		double newIxy =oldCell->iXY-(oldCell->volume )*(xcm*ycm)+(oldCell->volume-1)*newXCM*newYCM+ptTrans.x*ptTrans.y;

		double currLength = 4.0*sqrt(((float)((0.5*(oldCell->iXX + oldCell->iYY)) + .5*sqrt((float)((oldCell->iXX - oldCell->iYY)*(oldCell->iXX - oldCell->iYY) + 4*(oldCell->iXY)*(oldCell->iXY)))))/(float)(oldCell->volume));
		double currEnergy = lambdaLength * (currLength - targetLength)*(currLength - targetLength);

		double newLength;
		if(oldCell->volume<=1){
			newLength = 0.0;
		}else{
			newLength = 4.0*sqrt(((float)((0.5*(newIxx + newIyy)) + .5*sqrt((float)((newIxx - newIyy)*(newIxx - newIyy) + 4*newIxy*newIxy))))/(float)(oldCell->volume-1));
		}

		double newEnergy = lambdaLength * (newLength - targetLength) * (newLength - targetLength);

		energy += newEnergy - currEnergy;
	}


	if(energy!=energy)
		return 0.0;
	else
		return energy;
}


double LengthConstraintPlugin::changeEnergy_yz(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {


	// Assumption: COM and Volume has not been updated.

	/// E = lambda * (length - targetLength) ^ 2 

	//Center of mass, length constraints calculations are done withou checking whether cell volume reaches 0 or not
	// when cell is about to disappear this results in Nan values of energy - because division by 0 is involved or
	// sqrt(expression involving compoinents of inertia tensor) is NaN
	//in all such cases we set energy to 0 i.e. if energy=Nan we set it to energy=0.0

	double energy = 0.0;

	if (oldCell == newCell) return 0.0;

	Coordinates3D<double> ptTrans=boundaryStrategy->calculatePointCoordinates(pt);   
	//as in the original version 
	if (newCell){
		//local definitions of length constraint have priority over by type definitions
		double lambdaLength=lengthConstraintDataAccessor.get(newCell->extraAttribPtr)->lambdaLength;;
		double targetLength=lengthConstraintDataAccessor.get(newCell->extraAttribPtr)->targetLength;;
		
		if(lambdaLength==0.0 ) {
			if ( newCell->type < lengthEnergyParamVector.size() ){
				lambdaLength=lengthEnergyParamVector[newCell->type].lambdaLength;
				targetLength=lengthEnergyParamVector[newCell->type].targetLength;
			}
		}

		double ycm = (newCell->yCM / (float) newCell->volume);
		double zcm = (newCell->zCM / (float) newCell->volume);
		double newYCM = (newCell->yCM + ptTrans.y)/((float)newCell->volume + 1);
		double newZCM = (newCell->zCM + ptTrans.z)/((float)newCell->volume + 1);

		double newIyy=newCell->iYY+(newCell->volume )*zcm*zcm-(newCell->volume+1)*(newZCM*newZCM)+ptTrans.z*ptTrans.z;
		double newIzz=newCell->iZZ+(newCell->volume )*ycm*ycm-(newCell->volume+1)*(newYCM*newYCM)+ptTrans.y*ptTrans.y;
		double newIyz=newCell->iYZ-(newCell->volume )*ycm*zcm+(newCell->volume+1)*newYCM*newZCM-ptTrans.y*ptTrans.z;


		double currLength = 4.0*sqrt(((float)((0.5*(newCell->iYY + newCell->iZZ)) + .5*sqrt((float)((newCell->iYY - newCell->iZZ)*(newCell->iYY - newCell->iZZ) + 4*(newCell->iYZ)*(newCell->iYZ)))))/(float)(newCell->volume));

		double currEnergy = lambdaLength * (currLength - targetLength)*(currLength - targetLength);
		double newLength = 4.0*sqrt(((float)((0.5*(newIyy + newIzz)) + .5*sqrt((float)((newIyy - newIzz)*(newIyy - newIzz) + 4*newIyz*newIyz))))/(float)(newCell->volume+1));
		double newEnergy = lambdaLength * (newLength - targetLength)*(newLength - targetLength);
		energy += newEnergy - currEnergy;
	}
	if (oldCell) {
		//local definitions of length constraint have priority over by type definitions
		double lambdaLength=lengthConstraintDataAccessor.get(oldCell->extraAttribPtr)->lambdaLength;;
		double targetLength=lengthConstraintDataAccessor.get(oldCell->extraAttribPtr)->targetLength;;
		
		if(lambdaLength==0.0 ) {
			if ( oldCell->type < lengthEnergyParamVector.size() ){
				lambdaLength=lengthEnergyParamVector[oldCell->type].lambdaLength;
				targetLength=lengthEnergyParamVector[oldCell->type].targetLength;
			}
		}

		double ycm = (oldCell->yCM / (float) oldCell->volume);
		double zcm = (oldCell->zCM / (float) oldCell->volume);
		double newYCM = (oldCell->yCM - ptTrans.y)/((float)oldCell->volume - 1);
		double newZCM = (oldCell->zCM - ptTrans.z)/((float)oldCell->volume - 1);

		double newIyy =oldCell->iYY+(oldCell->volume )*(zcm*zcm)-(oldCell->volume-1)*(newZCM*newZCM)-ptTrans.z*ptTrans.z;
		double newIzz =oldCell->iZZ+(oldCell->volume )*(ycm*ycm)-(oldCell->volume-1)*(newYCM*newYCM)-ptTrans.y*ptTrans.y;
		double newIyz =oldCell->iYZ-(oldCell->volume )*(ycm*zcm)+(oldCell->volume-1)*newYCM*newZCM+ptTrans.y*ptTrans.z;



		double currLength = 4.0*sqrt(((float)((0.5*(oldCell->iYY + oldCell->iZZ)) + .5*sqrt((float)((oldCell->iYY - oldCell->iZZ)*(oldCell->iYY - oldCell->iZZ) + 4*(oldCell->iYZ)*(oldCell->iYZ)))))/(float)(oldCell->volume));

		double currEnergy = lambdaLength * (currLength - targetLength)*(currLength - targetLength);

		double newLength;
		if(oldCell->volume<=1){
			newLength = 0.0;
		}else{
			newLength = 4.0*sqrt(((float)((0.5*(newIyy + newIzz)) + .5*sqrt((float)((newIyy - newIzz)*(newIyy - newIzz) + 4*newIyz*newIyz))))/(float)(oldCell->volume-1));
		}    


		double newEnergy = lambdaLength * (newLength - targetLength) * (newLength - targetLength);
		energy += newEnergy - currEnergy;
	}

	if(energy!=energy)
		return 0.0;
	else
		return energy;

}


double LengthConstraintPlugin::changeEnergy_3D(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {

	// Assumption: COM and Volume has not been updated.

	/// E = lambda * (length - targetLength) ^ 2 

	//Center of mass, length constraints calculations are done withou checking whether cell volume reaches 0 or not
	// when cell is about to disappear this results in Nan values of energy - because division by 0 is involved or
	// sqrt(expression involving compoinents of inertia tensor) is NaN
	//in all such cases we set energy to 0 i.e. if energy=Nan we set it to energy=0.0

	double energy = 0.0;

	if (oldCell == newCell) return 0.0;

	Coordinates3D<double> ptTrans=boundaryStrategy->calculatePointCoordinates(pt);  

	//as in the original version 
	if (newCell){
		//local definitions of length constraint have priority over by type definitions
		double lambdaLength=lengthConstraintDataAccessor.get(newCell->extraAttribPtr)->lambdaLength;;
		double targetLength=lengthConstraintDataAccessor.get(newCell->extraAttribPtr)->targetLength;;
		double minorTargetLength=lengthConstraintDataAccessor.get(newCell->extraAttribPtr)->minorTargetLength;;

		if(lambdaLength==0.0 ) {
			if ( newCell->type < lengthEnergyParamVector.size() ){
				lambdaLength=lengthEnergyParamVector[newCell->type].lambdaLength;
				targetLength=lengthEnergyParamVector[newCell->type].targetLength;
				minorTargetLength=lengthEnergyParamVector[newCell->type].minorTargetLength;
			}
		}

		double xcm = (newCell->xCM / (float) newCell->volume);
		double ycm = (newCell->yCM / (float) newCell->volume);
		double zcm = (newCell->zCM / (float) newCell->volume);
		double newXCM = (newCell->xCM + ptTrans.x)/((float)newCell->volume + 1);
		double newYCM = (newCell->yCM + ptTrans.y)/((float)newCell->volume + 1);	 
		double newZCM = (newCell->zCM + ptTrans.z)/((float)newCell->volume + 1);	 

		double newIxx=newCell->iXX+(newCell->volume )*(ycm*ycm+zcm*zcm)-(newCell->volume+1)*(newYCM*newYCM+newZCM*newZCM)+ptTrans.y*ptTrans.y+ptTrans.z*ptTrans.z;
		double newIyy=newCell->iYY+(newCell->volume )*(xcm*xcm+zcm*zcm)-(newCell->volume+1)*(newXCM*newXCM+newZCM*newZCM)+ptTrans.x*ptTrans.x+ptTrans.z*ptTrans.z;
		double newIzz=newCell->iZZ+(newCell->volume )*(xcm*xcm+ycm*ycm)-(newCell->volume+1)*(newXCM*newXCM+newYCM*newYCM)+ptTrans.x*ptTrans.x+ptTrans.y*ptTrans.y;

		double newIxy=newCell->iXY-(newCell->volume )*xcm*ycm+(newCell->volume+1)*newXCM*newYCM-ptTrans.x*ptTrans.y;
		double newIxz=newCell->iXZ-(newCell->volume )*xcm*zcm+(newCell->volume+1)*newXCM*newZCM-ptTrans.x*ptTrans.z;
		double newIyz=newCell->iYZ-(newCell->volume )*ycm*zcm+(newCell->volume+1)*newYCM*newZCM-ptTrans.y*ptTrans.z;

		vector<double> aCoeff(4,0.0);
		vector<double> aCoeffNew(4,0.0);
		vector<complex<double> > roots;
		vector<complex<double> > rootsNew;

		//initialize coefficients of cubic eq used to find eigenvalues of inertia tensor - before pixel copy
		aCoeff[0]=-1.0;

		aCoeff[1]=newCell->iXX + newCell->iYY + newCell->iZZ;

		aCoeff[2]=newCell->iXY*newCell->iXY + newCell->iXZ*newCell->iXZ + newCell->iYZ*newCell->iYZ
			-newCell->iXX*newCell->iYY - newCell->iXX*newCell->iZZ - newCell->iYY*newCell->iZZ;

		aCoeff[3]=newCell->iXX*newCell->iYY*newCell->iZZ + 2*newCell->iXY*newCell->iXZ*newCell->iYZ
			-newCell->iXX*newCell->iYZ*newCell->iYZ
			-newCell->iYY*newCell->iXZ*newCell->iXZ
			-newCell->iZZ*newCell->iXY*newCell->iXY;

		roots=solveCubicEquationRealCoeeficients(aCoeff);


		//initialize coefficients of cubic eq used to find eigenvalues of inertia tensor - after pixel copy

		aCoeffNew[0]=-1.0;

		aCoeffNew[1]=newIxx + newIyy + newIzz;

		aCoeffNew[2]=newIxy*newIxy + newIxz*newIxz + newIyz*newIyz
			-newIxx*newIyy - newIxx*newIzz - newIyy*newIzz;

		aCoeffNew[3]=newIxx*newIyy*newIzz + 2*newIxy*newIxz*newIyz
			-newIxx*newIyz*newIyz
			-newIyy*newIxz*newIxz
			-newIzz*newIxy*newIxy;

		rootsNew=solveCubicEquationRealCoeeficients(aCoeffNew);


		//finding semiaxes of the ellipsoid
		//Ixx=m/5.0*(a_y^2+a_z^2) - andy cyclical permutations for other coordinate combinations
		//a_x,a_y,a_z are lengths of semiaxes of the allipsoid
		// We can invert above system of equations to get:
		vector<double> axes(3,0.0);

		axes[0]=sqrt((2.5/newCell->volume)*(roots[1].real()+roots[2].real()-roots[0].real()));
		axes[1]=sqrt((2.5/newCell->volume)*(roots[0].real()+roots[2].real()-roots[1].real()));
		axes[2]=sqrt((2.5/newCell->volume)*(roots[0].real()+roots[1].real()-roots[2].real()));

		//sorting semiaxes according the their lengths (shortest first)
		sort(axes.begin(),axes.end());



		vector<double> axesNew(3,0.0);

		axesNew[0]=sqrt((2.5/(newCell->volume+1))*(rootsNew[1].real()+rootsNew[2].real()-rootsNew[0].real()));
		axesNew[1]=sqrt((2.5/(newCell->volume+1))*(rootsNew[0].real()+rootsNew[2].real()-rootsNew[1].real()));
		axesNew[2]=sqrt((2.5/(newCell->volume+1))*(rootsNew[0].real()+rootsNew[1].real()-rootsNew[2].real()));

		//sorting semiaxes according the their lengths (shortest first)
		sort(axesNew.begin(),axesNew.end());

		// for (int i = 0 ; i < axesNew.size() ; ++i)
		// cerr<<"axesNew["<<i<<"]="<<axesNew[i]<<endl;

		// for (int i = 0 ;i<roots.size();++i){
		// cerr<<"root["<<i<<"]="<<roots[i]<<endl;
		// }
		// cerr<<"newCell->volume="<<newCell->volume<<" newCell->surface="<<newCell->surface<<endl;
		double currLength=2.0*axes[2];
		double currMinorLength=2.0*axes[0];
		// cerr<<" currLength="<<currLength<<" currMinorLength="<<currMinorLength<<endl;
		// cerr<<"minorTargetLength="<<lengthEnergyParamVector[newCell->type].minorTargetLength<<endl;

		double currEnergy = lambdaLength * ((currLength - targetLength)*(currLength - targetLength)+(currMinorLength - minorTargetLength)*(currMinorLength - minorTargetLength));

		double newLength = 2.0*axesNew[2];
		double newMinorLength=2.0*axesNew[0];

		double newEnergy = lambdaLength * ((newLength - targetLength)*(newLength - targetLength)+(newMinorLength - minorTargetLength)*(newMinorLength - minorTargetLength));
		energy += newEnergy - currEnergy;

		 //cerr<<"NEW energy="<<energy<<endl;
	}
	if (oldCell) {
		//cerr<<"****************OLD CELL PART***********************"<<endl;
		double lambdaLength=lengthConstraintDataAccessor.get(oldCell->extraAttribPtr)->lambdaLength;;
		double targetLength=lengthConstraintDataAccessor.get(oldCell->extraAttribPtr)->targetLength;;
		double minorTargetLength=lengthConstraintDataAccessor.get(oldCell->extraAttribPtr)->minorTargetLength;;

		if(lambdaLength==0.0 ) {
			if ( oldCell->type < lengthEnergyParamVector.size() ){
				lambdaLength=lengthEnergyParamVector[oldCell->type].lambdaLength;
				targetLength=lengthEnergyParamVector[oldCell->type].targetLength;
				minorTargetLength=lengthEnergyParamVector[oldCell->type].minorTargetLength;
			}
		}
		double xcm = (oldCell->xCM / (float) oldCell->volume);
		double ycm = (oldCell->yCM / (float) oldCell->volume);
		double zcm = (oldCell->zCM / (float) oldCell->volume);
		double newXCM = (oldCell->xCM - ptTrans.x)/((float)oldCell->volume - 1);
		double newYCM = (oldCell->yCM - ptTrans.y)/((float)oldCell->volume - 1);
		double newZCM = (oldCell->zCM - ptTrans.z)/((float)oldCell->volume - 1);

		double newIxx =oldCell->iXX+(oldCell->volume )*(ycm*ycm+zcm*zcm)-(oldCell->volume-1)*(newYCM*newYCM+newZCM*newZCM) - (ptTrans.y*ptTrans.y+ptTrans.z*ptTrans.z);
		double newIyy =oldCell->iYY+(oldCell->volume )*(xcm*xcm+zcm*zcm)-(oldCell->volume-1)*(newXCM*newXCM+newZCM*newZCM) - (ptTrans.x*ptTrans.x+ptTrans.z*ptTrans.z);
		double newIzz =oldCell->iZZ+(oldCell->volume )*(xcm*xcm+ycm*ycm)-(oldCell->volume-1)*(newXCM*newXCM+newYCM*newYCM) - (ptTrans.x*ptTrans.x+ptTrans.y*ptTrans.y);

		double newIxy =oldCell->iXY-(oldCell->volume )*(xcm*ycm)+(oldCell->volume-1)*newXCM*newYCM+ptTrans.x*ptTrans.y;
		double newIxz =oldCell->iXZ-(oldCell->volume )*(xcm*zcm)+(oldCell->volume-1)*newXCM*newZCM+ptTrans.x*ptTrans.z;
		double newIyz =oldCell->iYZ-(oldCell->volume )*(ycm*zcm)+(oldCell->volume-1)*newYCM*newZCM+ptTrans.y*ptTrans.z;


		vector<double> aCoeff(4,0.0);
		vector<double> aCoeffNew(4,0.0);
		vector<complex<double> > roots;
		vector<complex<double> > rootsNew;

		//initialize coefficients of cubic eq used to find eigenvalues of inertia tensor - before pixel copy
		aCoeff[0]=-1.0;

		aCoeff[1]=oldCell->iXX + oldCell->iYY + oldCell->iZZ;

		aCoeff[2]=oldCell->iXY*oldCell->iXY + oldCell->iXZ*oldCell->iXZ + oldCell->iYZ*oldCell->iYZ
			-oldCell->iXX*oldCell->iYY - oldCell->iXX*oldCell->iZZ - oldCell->iYY*oldCell->iZZ;

		aCoeff[3]=oldCell->iXX*oldCell->iYY*oldCell->iZZ + 2*oldCell->iXY*oldCell->iXZ*oldCell->iYZ
			-oldCell->iXX*oldCell->iYZ*oldCell->iYZ
			-oldCell->iYY*oldCell->iXZ*oldCell->iXZ
			-oldCell->iZZ*oldCell->iXY*oldCell->iXY;

		roots=solveCubicEquationRealCoeeficients(aCoeff);


		//initialize coefficients of cubic eq used to find eigenvalues of inertia tensor - after pixel copy

		aCoeffNew[0]=-1.0;

		aCoeffNew[1]=newIxx + newIyy + newIzz;

		aCoeffNew[2]=newIxy*newIxy + newIxz*newIxz + newIyz*newIyz
			-newIxx*newIyy - newIxx*newIzz - newIyy*newIzz;

		aCoeffNew[3]=newIxx*newIyy*newIzz + 2*newIxy*newIxz*newIyz
			-newIxx*newIyz*newIyz
			-newIyy*newIxz*newIxz
			-newIzz*newIxy*newIxy;

		rootsNew=solveCubicEquationRealCoeeficients(aCoeffNew);

		//finding semiaxes of the ellipsoid
		//Ixx=m/5.0*(a_y^2+a_z^2) - and cyclical permutations for other coordinate combinations
		//a_x,a_y,a_z are lengths of semiaxes of the allipsoid
		// We can invert above system of equations to get:
		vector<double> axes(3,0.0);

		axes[0]=sqrt((2.5/oldCell->volume)*(roots[1].real()+roots[2].real()-roots[0].real()));
		axes[1]=sqrt((2.5/oldCell->volume)*(roots[0].real()+roots[2].real()-roots[1].real()));
		axes[2]=sqrt((2.5/oldCell->volume)*(roots[0].real()+roots[1].real()-roots[2].real()));

		//sorting semiaxes according the their lengths (shortest first)
		sort(axes.begin(),axes.end());

		vector<double> axesNew(3,0.0);
		if (oldCell->volume<=1){
			axesNew[0]=0.0;
			axesNew[1]=0.0;
			axesNew[2]=0.0;
		}else{
			axesNew[0]=sqrt((2.5/(oldCell->volume-1))*(rootsNew[1].real()+rootsNew[2].real()-rootsNew[0].real()));
			axesNew[1]=sqrt((2.5/(oldCell->volume-1))*(rootsNew[0].real()+rootsNew[2].real()-rootsNew[1].real()));
			axesNew[2]=sqrt((2.5/(oldCell->volume-1))*(rootsNew[0].real()+rootsNew[1].real()-rootsNew[2].real()));
		}
		//sorting semiaxes according the their lengths (shortest first)
		sort(axesNew.begin(),axesNew.end());	

		double currLength = 2.0*axes[2];
		double currMinorLength=2.0*axes[0];

		//cerr<<"roots[1].real()+roots[2].real()-roots[0].real()="<<roots[1].real()+roots[2].real()-roots[0].real()<<endl;
		//cerr<<"rootsNew[1].real()+rootsNew[2].real()-rootsNew[0].real()="<<rootsNew[1].real()+rootsNew[2].real()-rootsNew[0].real()<<endl;

		//for (int i =0 ; i<3 ;++i){
		//	cerr<<"rootsNew["<<i<<"]="<<rootsNew[i]<<endl;			
		//}

		//for (int i =0 ; i<3 ;++i){
		//	cerr<<"axesNew["<<i<<"]="<<axesNew[i]<<endl;			
		//}

		double currEnergy = lambdaLength * ((currLength - targetLength)*(currLength - targetLength)+(currMinorLength - minorTargetLength)*(currMinorLength - minorTargetLength));
		//double currEnergy = lengthEnergyParamVector[oldCell->type].lambdaLength * (currLength - lengthEnergyParamVector[oldCell->type].targetLength)*(currLength - lengthEnergyParamVector[oldCell->type].targetLength);

		double newLength = 2.0*axesNew[2];
		double newMinorLength=2.0*axesNew[0];

		double newEnergy = lambdaLength * ((newLength - targetLength)*(newLength - targetLength)+(newMinorLength - minorTargetLength)*(newMinorLength - minorTargetLength));
		//double newEnergy = lengthEnergyParamVector[oldCell->type].lambdaLength * (newLength - lengthEnergyParamVector[oldCell->type].targetLength) * (newLength - lengthEnergyParamVector[oldCell->type].targetLength);

		energy += newEnergy - currEnergy;
		//cerr<<"lambdaLength="<<lambdaLength <<" targetLength="<<targetLength<<" minorTargetLength="<<minorTargetLength<<endl;
		
	}

	//cerr<<"energy="<<energy<<endl;
	if(energy!=energy)
		return 0.0;
	else
		return energy;
}


void LengthConstraintPlugin::initTypeId(Potts3D * potts){
	unsigned char maxType(0);
	Automaton * automaton=potts->getAutomaton();

	vector<unsigned char> typeIdVec(typeNameVec.size(),0);

	vector<LengthEnergyParam> lepVec=lengthEnergyParamVector;//temporaty storage
	//translate type name to type id
	for(unsigned int i =0 ; i < typeNameVec.size() ;++i){
		typeIdVec[i]=automaton->getTypeId(typeNameVec[i]);

		if(typeIdVec[i]>maxType)
			maxType=typeIdVec[i];
	}

	//assigning vector lambda targetVol pairs in such a wav that it will be possible to use e.g.vec[cellType].lambda statements
	// note that some of the vector elements migh be left default initialized
	lengthEnergyParamVector.clear();
	lengthEnergyParamVector.assign(maxType+1,LengthEnergyParam());

	for(unsigned int i =0 ; i < typeIdVec.size() ;++i){
		lengthEnergyParamVector[typeIdVec[i]]=lepVec[i];
	}

}



std::string LengthConstraintPlugin::toString(){
	return string("LengthConstraint");
}

std::string LengthConstraintPlugin::steerableName(){

	return toString();

}

