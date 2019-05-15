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
#include <CompuCell3D/plugins/ElasticityTracker/ElasticityTrackerPlugin.h>


#include <CompuCell3D/plugins/ElasticityTracker/ElasticityTracker.h>

using namespace CompuCell3D;


using namespace std;

#include "ElasticityPlugin.h"




ElasticityPlugin::ElasticityPlugin() : 
cellFieldG(0), 
pluginName("Elasticity"),
targetLengthElasticity(0.0),
lambdaElasticity(0.0),
maxLengthElasticity(100000000000.0),
diffEnergyFcnPtr(&ElasticityPlugin::diffEnergyGlobal),
boundaryStrategy(0),
simulator(0),
potts(0)
{}

ElasticityPlugin::~ElasticityPlugin() {

}

void ElasticityPlugin::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {
   simulator=_simulator;	
   potts = simulator->getPotts();
   cellFieldG = potts->getCellFieldG();

   pluginName=_xmlData->getAttribute("Name");

   potts->registerEnergyFunctionWithName(this,"ElasticityEnergy");
   simulator->registerSteerableObject(this);
   update(_xmlData,true);


}

void ElasticityPlugin::extraInit(Simulator *simulator) {

  cellFieldG = potts->getCellFieldG();
  
  fieldDim=cellFieldG ->getDim();
  boundaryStrategy=BoundaryStrategy::getInstance();
  

   bool pluginAlreadyRegisteredFlag;
   ElasticityTrackerPlugin * trackerPlugin = (ElasticityTrackerPlugin *) Simulator::pluginManager.get("ElasticityTracker",&pluginAlreadyRegisteredFlag); //this will load ElasticityTracker plugin if it is not already loaded  
   if(!pluginAlreadyRegisteredFlag)
      trackerPlugin->init(simulator);
   elasticityTrackerAccessorPtr=trackerPlugin->getElasticityTrackerAccessorPtr() ;

//   surfaceEnergy->setMaxNeighborIndex( trackerPlugin->getMaxNeighborIndex() );
   

}

void ElasticityPlugin::handleEvent(CC3DEvent & _event){
	if (_event.id!=LATTICE_RESIZE){
		return;
	}
	CC3DEventLatticeResize ev = static_cast<CC3DEventLatticeResize&>(_event);

	Dim3D shiftVec=ev.shiftVec;
	
    cellFieldG = potts->getCellFieldG();
	  
	fieldDim=cellFieldG ->getDim();


}

void ElasticityPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
    
	Potts3D *potts = simulator->getPotts();

	//if(potts->getDisplayUnitsFlag()){
	//	Unit targetLengthElasticityUnit=potts->getLengthUnit();
	//	Unit lambdaElasticityUnit=potts->getEnergyUnit()/(targetLengthElasticityUnit*targetLengthElasticityUnit);

	//	CC3DXMLElement * unitsElem=_xmlData->getFirstElement("Units"); 
	//	if (!unitsElem){ //add Units element
	//		unitsElem=_xmlData->attachElement("Units");
	//	}

	//	if(unitsElem->getFirstElement("TargetLengthElasticityUnit")){
	//		unitsElem->getFirstElement("TargetLengthElasticityUnit")->updateElementValue(targetLengthElasticityUnit.toString());
	//	}else{
	//		 unitsElem->attachElement("TargetLengthElasticityUnit",targetLengthElasticityUnit.toString());
	//	}



	//	if(unitsElem->getFirstElement("MaxElasticityLengthUnit")){
	//		unitsElem->getFirstElement("MaxElasticityLengthUnit")->updateElementValue(targetLengthElasticityUnit.toString());
	//	}else{
	//		unitsElem->attachElement("MaxElasticityLengthUnit",targetLengthElasticityUnit.toString());
	//	}




	//	if(unitsElem->getFirstElement("LambdaElasticityUnit")){
	//		unitsElem->getFirstElement("LambdaElasticityUnit")->updateElementValue(lambdaElasticityUnit.toString());
	//	}else{
	//		unitsElem->attachElement("LambdaElasticityUnit",lambdaElasticityUnit.toString());
	//	}

	//}

	if(_xmlData->findElement("Local")){
      diffEnergyFcnPtr=&ElasticityPlugin::diffEnergyLocal;
   }else{
      diffEnergyFcnPtr=&ElasticityPlugin::diffEnergyGlobal;
		if(_xmlData->findElement("TargetLengthElasticity"))
			targetLengthElasticity=_xmlData->getFirstElement("TargetLengthElasticity")->getDouble();
		if(_xmlData->findElement("LambdaElasticity"))
			lambdaElasticity=_xmlData->getFirstElement("LambdaElasticity")->getDouble();
		if(_xmlData->findElement("MaxElasticityLength"))
			maxLengthElasticity=_xmlData->getFirstElement("MaxElasticityLength")->getDouble();

   }
}



double ElasticityPlugin::diffEnergyGlobal(float _deltaL,float _lBefore,const ElasticityTrackerData * _elasticityTrackerData,const CellG *_cell){
	//cerr<<"USING ENERGY GLOBAL lambda="<<lambdaElasticity<<endl;
   if(_cell->volume>1){
		if(_lBefore<maxLengthElasticity){
			return lambdaElasticity*_deltaL*(2*(_lBefore-targetLengthElasticity)+_deltaL);
		}else{
			return 0.0;
		}
   }else{//after spin flip oldCell will disappear so the only contribution from before spin flip i.e. -(l-l0)^2
		if(_lBefore<maxLengthElasticity){
			return -lambdaElasticity*(_lBefore-targetLengthElasticity)*(_lBefore-targetLengthElasticity);
		}else{
			return 0.0;
		}
   }

}

double ElasticityPlugin::diffEnergyLocal(float _deltaL,float _lBefore,const ElasticityTrackerData * _elasticityTrackerData,const CellG *_cell){

   float lambdaLocal=_elasticityTrackerData->lambdaLength;
   float targetLengthLocal=_elasticityTrackerData->targetLength;
   float maxLengthElasticityLocal=_elasticityTrackerData->maxLengthElasticity;
   
   if(_cell->volume>1){
      if(_lBefore<maxLengthElasticityLocal){	
		//cerr<<"lambdaLocal="<<lambdaLocal<<endl;
	 //   cerr<<"targetLengthLocal="<<targetLengthLocal<<endl;
		//cerr<<"maxLengthElasticityLocal="<<maxLengthElasticityLocal<<endl;
		//cerr<<"returning energy local = "<<lambdaLocal*_deltaL*(2*(_lBefore-targetLengthLocal)+_deltaL)<<endl;
        return lambdaLocal*_deltaL*(2*(_lBefore-targetLengthLocal)+_deltaL);
      }else{
		//cerr<<"returning energy local = 0.0"<<endl;
        return 0.0;
      }
      
   }else{//after spin flip oldCell will disappear so the only contribution from before spin flip i.e. -(l-l0)^2
      if(_lBefore<maxLengthElasticityLocal){
		//cerr<<"1 lambdaLocal="<<lambdaLocal<<endl;
		//cerr<<"1 targetLengthLocal="<<targetLengthLocal<<endl;
		//cerr<<"1 maxLengthElasticityLocal="<<maxLengthElasticityLocal<<endl;
	 //   cerr<<"1 returning energy local = "<<-lambdaLocal*(_lBefore-targetLengthLocal)*(_lBefore-targetLengthLocal)<<endl;
        return -lambdaLocal*(_lBefore-targetLengthLocal)*(_lBefore-targetLengthLocal);
      }else{
		//cerr<<"returning energy local = 0.0"<<endl;
        return 0.0;
      }
   }

}



double ElasticityPlugin::changeEnergy(const Point3D &pt,
				  const CellG *newCell,
				  const CellG *oldCell) {

   
//    //Change in Energy is given by E_after-E_before
//    //((l+d)-l0)^2-(l-l0)^2 = d*(2*(l-l0)+d)
   float energy=0.0;
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
         centroidOldAfter=precalculateCentroid(pt, oldCell, -1,fieldDim,boundaryStrategy);
         centMassOldAfter.XRef()=centroidOldAfter.X()/(float)(oldCell->volume-1);
         centMassOldAfter.YRef()=centroidOldAfter.Y()/(float)(oldCell->volume-1);
         centMassOldAfter.ZRef()=centroidOldAfter.Z()/(float)(oldCell->volume-1);
         
      }else{
//          return 0.0;//if oldCell is to disappear the Elasticity energy will be zero
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

   
   //will loop over neighbors of the oldCell and calculate Elasticity energy
   set<ElasticityTrackerData> * elasticityNeighborsTmpPtr;
   set<ElasticityTrackerData>::iterator sitr;
   CellG *nCell;
   float deltaL;
   float lBefore;
   float oldVol;
   float newVol;
   float nCellVol;
   if(oldCell){
      oldVol=oldCell->volume;
      elasticityNeighborsTmpPtr=&elasticityTrackerAccessorPtr->get(oldCell->extraAttribPtr)->elasticityNeighbors ;
      
      for (sitr=elasticityNeighborsTmpPtr->begin() ; sitr != elasticityNeighborsTmpPtr->end() ;++sitr){
         nCell=sitr->neighborAddress;
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
		 //cerr<<"centMassOldBefore="<<centMassOldBefore<<endl;
		 //cerr<<"centMassOldAfter="<centMassOldAfter<<endl;	
		 //cerr<<"deltaL="<<deltaL<<" lBefore="<<lBefore<<endl;
         energy+=(this->*diffEnergyFcnPtr)(deltaL,lBefore,&(*sitr),oldCell);

//          if(oldCell->volume>1){
//             energy+=lambdaElasticity*deltaL*(2*(lBefore-targetLengthElasticity)+deltaL);
//          }else{//after spin flip oldCell will disappear so the only contribution from before spin flip i.e. -(l-l0)^2
//             energy-=lambdaElasticity*(lBefore-targetLengthElasticity)*(lBefore-targetLengthElasticity);
//          }
//          double locEn1;
//          double locEn2;
//          locEn2=(this->*diffEnergyFcnPtr)(deltaL,lBefore,&(*sitr),oldCell);
//          if(oldCell->volume>1){
//             locEn1=lambdaElasticity*deltaL*(2*(lBefore-targetLengthElasticity)+deltaL);
//          }else{//after spin flip oldCell will disappear so the only contribution from before spin flip i.e. -(l-l0)^2
//             locEn1=-lambdaElasticity*(lBefore-targetLengthElasticity)*(lBefore-targetLengthElasticity);
//          }
// 
//          if(locEn1!=locEn2){
//             cerr<<"locEn1="<<locEn1<<" locEn2="<<locEn2<<endl;
//             exit(0);
// 
//          }

      }
   }

   if(newCell){
      newVol=newCell->volume;
      elasticityNeighborsTmpPtr=&elasticityTrackerAccessorPtr->get(newCell->extraAttribPtr)->elasticityNeighbors ;
      for (sitr=elasticityNeighborsTmpPtr->begin() ; sitr != elasticityNeighborsTmpPtr->end() ;++sitr){
         nCell=sitr->neighborAddress;
         nCellVol=nCell->volume;
         
         if(nCell!=oldCell){
            lBefore=distInvariantCM(centMassNewBefore.X(),centMassNewBefore.Y(),centMassNewBefore.Z(),nCell->xCM/nCellVol,nCell->yCM/nCellVol,nCell->zCM/nCellVol,fieldDim,boundaryStrategy);
            deltaL=
            distInvariantCM(centMassNewAfter.X(),centMassNewAfter.Y(),centMassNewAfter.Z(),nCell->xCM/nCellVol,nCell->yCM/nCellVol,nCell->zCM/nCellVol,fieldDim,boundaryStrategy)
            -lBefore;
         }else{// this was already taken into account in the oldCell secion - we need to avoid double counting
//             lBefore=distInvariantCM(centMassNewBefore.X(),centMassNewBefore.Y(),centMassNewBefore.Z(),centMassOldBefore.X(),centMassOldBefore.Y(),centMassOldBefore.Z(),fieldDim);
//             deltaL=
//             distInvariantCM(centMassNewAfter.X(),centMassNewAfter.Y(),centMassNewAfter.Z(),centMassOldAfter.X(),centMassOldAfter.Y(),centMassOldAfter.Z(),fieldDim)
//             -lBefore;

         }
         energy+=(this->*diffEnergyFcnPtr)(deltaL,lBefore,&(*sitr),newCell);
//          energy+=lambdaElasticity*deltaL*(2*(lBefore-targetLengthElasticity)+deltaL);

//          double locEn1;
//          double locEn2;
//          locEn2=(this->*diffEnergyFcnPtr)(deltaL,lBefore,&(*sitr),newCell);
//          
//          locEn1=lambdaElasticity*deltaL*(2*(lBefore-targetLengthElasticity)+deltaL);
//             cerr<<"locEn1="<<locEn1<<" locEn1="<<locEn2<<endl;
// 
//          if(locEn1!=locEn2){
//             cerr<<"locEn1="<<locEn1<<" locEn2="<<locEn2<<endl;
//             exit(0);
// 
//          }

      }
   }



   Coordinates3D<int> centroid;
//    if(oldCell){
// //        centroid=precalculateCMAfterFlip(pt, oldCell, -1,fieldDim);
//          centroid=precalculateCentroid(pt, oldCell, -1,fieldDim);
// 
// //       cerr<<"int="<<precalculateCentroid(pt, oldCell, -1,fieldDim)<<endl;
//       cerr<<"pt="<<pt<<endl;
//       cerr<<"oldCell xCM="<<oldCell->xCM<<" xcm="<<oldCell->xCM/(float)oldCell->volume<<endl;
//       cerr<<"Centroid "<<centroid.X()<<","<<centroid.Y()<<","<<centroid.Z()<<endl;
//       cerr<<"Manual "<<oldCell->xCM-pt.x<<","<<oldCell->yCM-pt.y<<","<<oldCell->zCM-pt.z<<endl;
//       if(oldCell->xCM-pt.x - centroid.X() !=0 || oldCell->yCM-pt.y - centroid.Y() !=0 || oldCell->zCM-pt.z - centroid.Z() !=0)
//          exit(0);
//    }


//    float energy=0.0;

//    cerr<<"energy="<<energy<<endl;
   return energy;


}

std::string ElasticityPlugin::toString(){
   return pluginName;
   //return "Elasticity";
}


std::string ElasticityPlugin::steerableName(){
   return toString();
}
