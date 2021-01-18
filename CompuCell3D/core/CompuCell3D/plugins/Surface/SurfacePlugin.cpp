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
#include <CompuCell3D/plugins/SurfaceTracker/SurfaceTrackerPlugin.h>

using namespace std;

#include "SurfacePlugin.h"


// SurfacePlugin::SurfacePlugin() : potts(0) {}

SurfacePlugin::~SurfacePlugin() {}

void SurfacePlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData){
	potts = simulator->getPotts();
	cellFieldG = (WatchableField3D<CellG *>*)potts->getCellFieldG();



	bool pluginAlreadyRegisteredFlag;
	SurfaceTrackerPlugin *plugin=(SurfaceTrackerPlugin*)Simulator::pluginManager.get("SurfaceTracker",&pluginAlreadyRegisteredFlag); //this will load SurfaceTracker plugin if it is not already loaded
	cerr<<"GOT HERE BEFORE CALLING INIT"<<endl;
	if(!pluginAlreadyRegisteredFlag)
		plugin->init(simulator);

	pUtils=simulator->getParallelUtils();

	pluginName=_xmlData->getAttribute("Name");

	maxNeighborIndex=plugin->getMaxNeighborIndex();
	lmf=plugin->getLatticeMultiplicativeFactors();

	potts->registerEnergyFunctionWithName(this,toString());
	//save pointer to plugin xml element for later. Initialization has to be done in extraInit to make sure sutomaton (CelltypePlugin)
	// is already registered - we need it in the case of BYCELLTYPE
	xmlData=_xmlData;
	
	simulator->registerSteerableObject(this);

}

void SurfacePlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

	//if(potts->getDisplayUnitsFlag()){
	//	Unit targetSurfaceUnit=powerUnit(potts->getLengthUnit(),2);
	//	Unit lambdaSurfaceUnit=potts->getEnergyUnit()/(targetSurfaceUnit*targetSurfaceUnit);

	//	CC3DXMLElement * unitsElem=_xmlData->getFirstElement("Units"); 
	//	if (!unitsElem){ //add Units element
	//		unitsElem=_xmlData->attachElement("Units");
	//	}

	//	if(unitsElem->getFirstElement("TargetSurfaceUnit")){
	//		unitsElem->getFirstElement("TargetSurfaceUnit")->updateElementValue(targetSurfaceUnit.toString());
	//	}else{
	//		CC3DXMLElement * surfaceUnitElem = unitsElem->attachElement("TargetSurfaceUnit",targetSurfaceUnit.toString());
	//	}

	//	if(unitsElem->getFirstElement("LambdaSurfaceUnit")){
	//		unitsElem->getFirstElement("LambdaSurfaceUnit")->updateElementValue(lambdaSurfaceUnit.toString());
	//	}else{
	//		CC3DXMLElement * lambdaSurfaceUnitElem = unitsElem->attachElement("LambdaSurfaceUnit",lambdaSurfaceUnit.toString());
	//	}


	//}


	if (_xmlData->findElement("SurfaceEnergyExpression")){
		unsigned int maxNumberOfWorkNodes=pUtils->getMaxNumberOfWorkNodesPotts();
		eed.allocateSize(maxNumberOfWorkNodes);
		vector<string> variableNames;
		variableNames.push_back("LambdaSurface");
		variableNames.push_back("Surface");
		variableNames.push_back("Starget");

		eed.addVariables(variableNames.begin(),variableNames.end());
		eed.update(_xmlData->getFirstElement("SurfaceEnergyExpression"));			
		energyExpressionDefined=true;
	}else{
		energyExpressionDefined=false;
	}

	//if there are no child elements for this plugin it means will use changeEnergyByCellId
	if(!_xmlData->findElement("SurfaceEnergyParameters") && !_xmlData->findElement("TargetSurface")){ 
		functionType=BYCELLID;
	}else{
		if(_xmlData->findElement("SurfaceEnergyParameters"))
			functionType=BYCELLTYPE;
		else if (_xmlData->findElement("TargetSurface"))
			functionType=GLOBAL;
		else //in case users put garbage xml use changeEnergyByCellId
			functionType=BYCELLID;
	}
	Automaton *automaton=potts->getAutomaton();
	cerr<<"automaton="<<automaton<<endl;

	switch(functionType){
		case BYCELLID:
			//set fcn ptr
			changeEnergyFcnPtr=&SurfacePlugin::changeEnergyByCellId;
			break;

		case BYCELLTYPE:
			{
				surfaceEnergyParamVector.clear();
				vector<int> typeIdVec;
				vector<SurfaceEnergyParam> surfaceEnergyParamVectorTmp;

				CC3DXMLElementList energyVec=_xmlData->getElements("SurfaceEnergyParameters");

				for (int i = 0 ; i<energyVec.size(); ++i){
					SurfaceEnergyParam surParam;

					surParam.targetSurface=energyVec[i]->getAttributeAsDouble("TargetSurface");
					surParam.lambdaSurface=energyVec[i]->getAttributeAsDouble("LambdaSurface");
					surParam.typeName=energyVec[i]->getAttribute("CellType");					
					typeIdVec.push_back(automaton->getTypeId(surParam.typeName));

					surfaceEnergyParamVectorTmp.push_back(surParam);				
				}
				vector<int>::iterator pos=max_element(typeIdVec.begin(),typeIdVec.end());
				int maxTypeId=*pos;
				surfaceEnergyParamVector.assign(maxTypeId+1,SurfaceEnergyParam());
				for (int i = 0 ; i < surfaceEnergyParamVectorTmp.size() ; ++i){
					surfaceEnergyParamVector[typeIdVec[i]]=surfaceEnergyParamVectorTmp[i];
				}

				//set fcn ptr
				changeEnergyFcnPtr=&SurfacePlugin::changeEnergyByCellType;
			}
			break;

		case GLOBAL:
			//using Global Surface Energy Parameters
			targetSurface=_xmlData->getFirstElement("TargetSurface")->getDouble();
			lambdaSurface=_xmlData->getFirstElement("LambdaSurface")->getDouble();
			
			//set fcn ptr
			changeEnergyFcnPtr=&SurfacePlugin::changeEnergyGlobal;
			break;

		default:
			//set fcn ptr
			changeEnergyFcnPtr=&SurfacePlugin::changeEnergyByCellId;
	}
	//check if there is a ScaleSurface parameter  in XML
	if(_xmlData->findElement("ScaleSurface")){
		scaleSurface=_xmlData->getFirstElement("ScaleSurface")->getDouble();
	}

	boundaryStrategy=BoundaryStrategy::getInstance();
}


void SurfacePlugin::extraInit(Simulator *simulator){
	update(xmlData);
}

void SurfacePlugin::handleEvent(CC3DEvent & _event){
    if (_event.id==CHANGE_NUMBER_OF_WORK_NODES){    
    	update(xmlData);
    }

}


std::pair<double,double> SurfacePlugin::getNewOldSurfaceDiffs(const Point3D &pt, const CellG *newCell,const CellG *oldCell){
	

	CellG *nCell;
   double oldDiff = 0.;
   double newDiff = 0.;
   Neighbor neighbor;
   for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
      neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
      if(!neighbor.distance){
      //if distance is 0 then the neighbor returned is invalid
      continue;
      }
      nCell = cellFieldG->get(neighbor.pt);
      if (newCell == nCell) newDiff-=lmf.surfaceMF;
      else newDiff+=lmf.surfaceMF;

      if (oldCell == nCell) oldDiff+=lmf.surfaceMF;
      else oldDiff-=lmf.surfaceMF;

   }	
	return make_pair(newDiff,oldDiff);
}

double SurfacePlugin::diffEnergy(double lambda, double targetSurface,double surface,  double diff) {
	if (!energyExpressionDefined){

		return lambda *(diff*diff + 2 * diff * (surface - fabs(targetSurface)));
	}else{
		int currentWorkNodeNumber=pUtils->getCurrentWorkNodeNumber();	
		ExpressionEvaluator & ev=eed[currentWorkNodeNumber];
		double energyBefore=0.0,energyAfter=0.0;

		//before
		ev[0]=lambda;
		ev[1]=surface;
		ev[2]=targetSurface;
		energyBefore=ev.eval();

		//after
		ev[1]=surface+diff;
		
		energyAfter=ev.eval();

		return energyAfter-energyBefore;
		
	}
}

double SurfacePlugin::changeEnergyGlobal(const Point3D &pt, const CellG *newCell,const CellG *oldCell){

	double energy = 0.0;

  if (oldCell == newCell) return 0.0;
  
  pair<double,double> newOldDiffs=getNewOldSurfaceDiffs(pt,newCell,oldCell);

  if (newCell){
    energy += diffEnergy(lambdaSurface , targetSurface , newCell->surface*scaleSurface, newOldDiffs.first * scaleSurface);

   }
  if (oldCell){
	 energy += diffEnergy(lambdaSurface , targetSurface , oldCell->surface*scaleSurface, newOldDiffs.second*scaleSurface);
  }
   
  
//       cerr<<"Surface Global Energy="<<energy<<endl;
  return energy;

}

double SurfacePlugin::changeEnergyByCellType(const Point3D &pt,
				  const CellG *newCell,
				  const CellG *oldCell) {

  /// E = lambda * (surface - targetSurface) ^ 2 
  
	double energy = 0.0;

  if (oldCell == newCell) return 0.0;
  
  pair<double,double> newOldDiffs=getNewOldSurfaceDiffs(pt,newCell,oldCell);

  if (newCell){
    energy += diffEnergy(surfaceEnergyParamVector[newCell->type].lambdaSurface , surfaceEnergyParamVector[newCell->type].targetSurface , newCell->surface*scaleSurface, newOldDiffs.first * scaleSurface);

   }
  if (oldCell){
	 energy += diffEnergy(surfaceEnergyParamVector[oldCell->type].lambdaSurface , surfaceEnergyParamVector[oldCell->type].targetSurface , oldCell->surface*scaleSurface, newOldDiffs.second*scaleSurface);
  }
   
  
//       cerr<<"Surface By Type Energy="<<energy<<endl;
  return energy;

}


double SurfacePlugin::changeEnergyByCellId(const Point3D &pt,
				  const CellG *newCell,
				  const CellG *oldCell) {

  /// E = lambda * (surface - targetSurface) ^ 2 
  
	double energy = 0.0;

  if (oldCell == newCell) return 0.0;
  
  pair<double,double> newOldDiffs=getNewOldSurfaceDiffs(pt,newCell,oldCell);

  if (newCell){
    energy += diffEnergy(newCell->lambdaSurface , newCell->targetSurface , newCell->surface*scaleSurface, newOldDiffs.first * scaleSurface);

   }
  if (oldCell){
	 energy += diffEnergy(oldCell->lambdaSurface , oldCell->targetSurface , oldCell->surface*scaleSurface, newOldDiffs.second*scaleSurface);
  }
   
  
//       cerr<<"Surface By Id Energy="<<energy<<endl;
  return energy;}




double SurfacePlugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {

	/// E = lambda * (surface - targetSurface) ^ 2 
	return (this->*changeEnergyFcnPtr)(pt,newCell,oldCell);


}


std::string SurfacePlugin::steerableName(){
	return pluginName;
	//return "Surface";
}

std::string SurfacePlugin::toString(){
	return pluginName;
	//return "Surface";
}




