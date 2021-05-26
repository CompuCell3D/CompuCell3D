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

#include "VolumePlugin.h"

VolumePlugin::~VolumePlugin() {}

void VolumePlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData){
	potts = simulator->getPotts();
	bool pluginAlreadyRegisteredFlag;
	Plugin *plugin=Simulator::pluginManager.get("VolumeTracker",&pluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded


	pUtils=simulator->getParallelUtils();
	pluginName=_xmlData->getAttribute("Name");

	cerr<<"GOT HERE BEFORE CALLING INIT"<<endl;
	if(!pluginAlreadyRegisteredFlag)
		plugin->init(simulator);
	potts->registerEnergyFunctionWithName(this,toString());
	//save pointer to plugin xml element for later. Initialization has to be done in extraInit to make sure sutomaton (CelltypePlugin)
	// is already registered - we need it in the case of BYCELLTYPE
	xmlData=_xmlData;

	simulator->registerSteerableObject(this);
        

}

void VolumePlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
	//if there are no child elements for this plugin it means will use changeEnergyByCellId

	if (_xmlData->findElement("VolumeEnergyExpression")){
		unsigned int maxNumberOfWorkNodes=pUtils->getMaxNumberOfWorkNodesPotts();
		eed.allocateSize(maxNumberOfWorkNodes);
		vector<string> variableNames;
		variableNames.push_back("LambdaVolume");
		variableNames.push_back("Volume");
		variableNames.push_back("Vtarget");

		eed.addVariables(variableNames.begin(),variableNames.end());
		eed.update(_xmlData->getFirstElement("VolumeEnergyExpression"));			
		energyExpressionDefined=true;
	}else{
		energyExpressionDefined=false;
	}


	if(!_xmlData->findElement("VolumeEnergyParameters") && !_xmlData->findElement("TargetVolume")){ 

		functionType=BYCELLID;
	}else{
		if(_xmlData->findElement("VolumeEnergyParameters"))
			functionType=BYCELLTYPE;
		else if (_xmlData->findElement("TargetVolume"))
			functionType=GLOBAL;
		else //in case users put garbage xml use changeEnergyByCellId
			functionType=BYCELLID;
	}
	Automaton *automaton=potts->getAutomaton();
	cerr<<"automaton="<<automaton<<endl;

	switch(functionType){
		case BYCELLID:
			//set fcn ptr
			changeEnergyFcnPtr=&VolumePlugin::changeEnergyByCellId;
			break;

		case BYCELLTYPE:
			{
				volumeEnergyParamMap.clear();

				CC3DXMLElementList energyVec=_xmlData->getElements("VolumeEnergyParameters");

				for (int i = 0 ; i<energyVec.size(); ++i){
					VolumeEnergyParam volParam;

					volParam.targetVolume=energyVec[i]->getAttributeAsDouble("TargetVolume");
					volParam.lambdaVolume=energyVec[i]->getAttributeAsDouble("LambdaVolume");
					volParam.typeName=energyVec[i]->getAttribute("CellType");
					volumeEnergyParamMap[automaton->getTypeId(volParam.typeName)] = volParam;
				}

				//set fcn ptr
				changeEnergyFcnPtr=&VolumePlugin::changeEnergyByCellType;
			}
			break;

		case GLOBAL:
			//using Global Volume Energy Parameters
			targetVolume=_xmlData->getFirstElement("TargetVolume")->getDouble();
			lambdaVolume=_xmlData->getFirstElement("LambdaVolume")->getDouble();			
			//set fcn ptr
			changeEnergyFcnPtr=&VolumePlugin::changeEnergyGlobal;
			break;

		default:
			//set fcn ptr
			changeEnergyFcnPtr=&VolumePlugin::changeEnergyByCellId;
	}
}


void VolumePlugin::extraInit(Simulator *simulator){
	update(xmlData);
}

void VolumePlugin::handleEvent(CC3DEvent & _event){
    if (_event.id==CHANGE_NUMBER_OF_WORK_NODES){    
    	update(xmlData);
    }

}
double VolumePlugin::customExpressionFunction(double _lambdaVolume,double _targetVolume, double _volumeBefore,double _volumeAfter){

	int currentWorkNodeNumber=pUtils->getCurrentWorkNodeNumber();	
	ExpressionEvaluator & ev=eed[currentWorkNodeNumber];
	double energyBefore=0.0,energyAfter=0.0;

	//before
	ev[0]=_lambdaVolume;
	ev[1]=_volumeBefore;
	ev[2]=_targetVolume;
	energyBefore=ev.eval();

	//after		
	ev[1]=_volumeAfter;		
	energyAfter=ev.eval();

	return energyAfter-energyBefore;
}

double VolumePlugin::changeEnergyGlobal(const Point3D &pt, const CellG *newCell,const CellG *oldCell){

	double energy = 0;

	if (oldCell == newCell) return 0;

	if (!energyExpressionDefined){
		//as in the original version 
		if (newCell){
			energy += lambdaVolume *
				(1 + 2 * (newCell->volume - targetVolume));

		}
		if (oldCell){
			energy += lambdaVolume *
				(1 - 2 * (oldCell->volume - targetVolume));


		}

		return energy;
	}else{

		if (newCell){
			energy+=customExpressionFunction(lambdaVolume,targetVolume,newCell->volume,newCell->volume+1);
		}
		
		if (oldCell){
			energy+=customExpressionFunction(lambdaVolume,targetVolume,oldCell->volume,oldCell->volume-1);

		}		
		return energy;
		

	}


}

double VolumePlugin::changeEnergyByCellType(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {

	/// E = lambda * (volume - targetVolume) ^ 2 

	double energy = 0;

	if (oldCell == newCell) return 0;

	if (!energyExpressionDefined){
		if (newCell)
			energy += volumeEnergyParamMap[newCell->type].lambdaVolume *
			(1 + 2 * (newCell->volume - fabs(volumeEnergyParamMap[newCell->type].targetVolume)));

		if (oldCell)
			energy += volumeEnergyParamMap[oldCell->type].lambdaVolume  *
			(1 - 2 * (oldCell->volume - fabs(volumeEnergyParamMap[oldCell->type].targetVolume)));


		//cerr<<"VOLUME CHANGE ENERGY NEW: "<<energy<<endl;
		return energy;


	}else{

		if (newCell){
			energy+=customExpressionFunction(volumeEnergyParamMap[newCell->type].lambdaVolume,fabs(volumeEnergyParamMap[newCell->type].targetVolume),newCell->volume,newCell->volume+1);
		}

		if (oldCell){
			energy+=customExpressionFunction(volumeEnergyParamMap[oldCell->type].lambdaVolume,fabs(volumeEnergyParamMap[oldCell->type].targetVolume),oldCell->volume,oldCell->volume-1);

		}		
		return energy;


	}


}


double VolumePlugin::changeEnergyByCellId(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {

	/// E = lambda * (volume - targetVolume) ^ 2 

	double energy = 0;

	if (oldCell == newCell) return 0;

	if (!energyExpressionDefined){

		if (newCell){

			energy +=  newCell->lambdaVolume*
				(1 + 2 * ((int)newCell->volume - newCell->targetVolume));
		}
		if (oldCell){
			energy += oldCell->lambdaVolume*
				(1 - 2 * ((int)oldCell->volume - oldCell->targetVolume));
		}



		return energy;


	}else{

		if (newCell){
			energy+=customExpressionFunction(newCell->lambdaVolume,newCell->targetVolume,newCell->volume,newCell->volume+1);
		}

		if (oldCell){
			energy+=customExpressionFunction(oldCell->lambdaVolume,oldCell->targetVolume,oldCell->volume,oldCell->volume-1);

		}		
		return energy;


	}


}




double VolumePlugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {

	/// E = lambda * (volume - targetVolume) ^ 2 
	return (this->*changeEnergyFcnPtr)(pt,newCell,oldCell);


}


std::string VolumePlugin::steerableName(){
	return pluginName;
	//return "Volume";
}

std::string VolumePlugin::toString(){
	return pluginName;
	//return "Volume";
}
