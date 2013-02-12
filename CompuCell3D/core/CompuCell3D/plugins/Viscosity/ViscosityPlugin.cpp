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

#include "ViscosityPlugin.h"


// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Automaton/Automaton.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>
// // // #include <PublicUtilities/NumericalUtils.h>
using namespace CompuCell3D;


// // // #include <BasicUtils/BasicString.h>
// // // #include <BasicUtils/BasicException.h>
// #include <CompuCell3D/plugins/CellVelocity/CellVelocityPlugin.h>
#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>




ViscosityPlugin::ViscosityPlugin():potts(0),sim(0),neighborTrackerAccessorPtr(0),lambdaViscosity(0),maxNeighborIndex(0)   {
}

ViscosityPlugin::~ViscosityPlugin() {  
}

double ViscosityPlugin::dist(double _x, double _y, double _z){
	return sqrt(_x*_x+_y*_y+_z*_z);
}

void ViscosityPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData){
	potts = simulator->getPotts();
	sim=simulator;

	bool pluginAlreadyRegisteredFlagCOM;
	Plugin *pluginCOM=Simulator::pluginManager.get("CenterOfMass",&pluginAlreadyRegisteredFlagCOM); //this will load CenterOFMass plugin if it is not already loaded

	bool pluginAlreadyRegisteredFlag;
	Plugin *plugin=Simulator::pluginManager.get("NeighborTracker",&pluginAlreadyRegisteredFlag); //this will load NeighborTracker plugin if it is not already loaded




	pluginName=_xmlData->getAttribute("Name");


	if(!pluginAlreadyRegisteredFlag)
		plugin->init(simulator);
	potts->registerEnergyFunctionWithName(this,toString());
	//save pointer to plugin xml element for later. Initialization has to be done in extraInit to make sure sutomaton (CelltypePlugin)
	// is already registered - we need it in the case of BYCELLTYPE
	xmlData=_xmlData;

	simulator->registerSteerableObject(this);

	//potts->registerCellGChangeWatcher(this);




	boundaryStrategy=BoundaryStrategy::getInstance();    
	potts->getBoundaryXName()=="Periodic" ? boundaryConditionIndicator.x=1 : boundaryConditionIndicator.x=0 ;
	potts->getBoundaryYName()=="Periodic" ? boundaryConditionIndicator.y=1 : boundaryConditionIndicator.y=0;
	potts->getBoundaryZName()=="Periodic" ? boundaryConditionIndicator.z=1 : boundaryConditionIndicator.z=0;

	fieldDim=potts->getCellFieldG()->getDim();

	maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);
}

void ViscosityPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

	lambdaViscosity=_xmlData->getFirstElement("LambdaViscosity")->getDouble();
	//cerr<<"lambdaViscosity="<<lambdaViscosity<<endl;
	//exit(0);

}

// void CompuCell3D::ViscosityPlugin::field3DChange(const Point3D &pt, CellG *newCell,CellG * oldCell){
// //COM field3DChange is called after this function so we copy COM  befiore it gets updated...
// }

void ViscosityPlugin::extraInit(Simulator *simulator) {
	// CellVelocityPlugin * cellVelocityPluginPtr=(CellVelocityPlugin*)(Simulator::pluginManager.get("CellInstantVelocity"));
	// ASSERT_OR_THROW("CellVelocity plugin not initialized!", cellVelocityPluginPtr);
	// ASSERT_OR_THROW("CellVelocityAccessorPtr  not initialized!", cellVelocityPluginPtr->getCellVelocityDataAccessorPtr());
	// ASSERT_OR_THROW("CellVelocityData: cldeque tas too small size - size=2 or greater needed!",
	// cellVelocityPluginPtr->getCldequeCapacity()>=2);

	// viscosityEnergy->setCellVelocityDataAccessorPtr(cellVelocityPluginPtr->getCellVelocityDataAccessorPtr());

	// CenterOfMassPlugin * centerOfMassPluginPtr=(CenterOfMassPlugin*)(Simulator::pluginManager.get("CenterOfMass"));
	// ASSERT_OR_THROW("CenterOfMass plugin not initialized!", centerOfMassPluginPtr);


	// viscosityEnergy->setCOMPtr(centerOfMassPluginPtr);

	// NeighborTrackerPlugin * neighborTrackerPluginPtr=(NeighborTrackerPlugin*)(Simulator::pluginManager.get("NeighborTracker"));
	// ASSERT_OR_THROW("NeighborTracker plugin not initialized!", neighborTrackerPluginPtr);
	// ASSERT_OR_THROW("neighborAccessorPtr  not initialized!", neighborTrackerPluginPtr->getNeighborTrackerAccessorPtr());
	update(xmlData);
	bool pluginAlreadyRegisteredFlag;
	NeighborTrackerPlugin *nTrackerPlugin=(NeighborTrackerPlugin*)Simulator::pluginManager.get("NeighborTracker",&pluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded

	neighborTrackerAccessorPtr=nTrackerPlugin->getNeighborTrackerAccessorPtr();

	//viscosityEnergy->initializeViscosityEnergy();
}

double ViscosityPlugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {
	if(sim->getStep()<100){
		return 0;
	}

	double energy = 0;
	unsigned int token = 0;
	double distance = 0;
	Point3D n;
	double cellDistance=0.0;
	double commonArea=0.0;
	double x0,y0,z0,x1,y1,z1;
	double velocityDiffX=0;
	double velocityDiffY=0;
	double velocityDiffZ=0;
	Coordinates3D<double> nCellCom0,nCellCom1,cellCom0,cellCom1;

	Coordinates3D<double> oldCellCMBefore, oldCellCMBeforeBefore, oldCellCMAfter, newCellCMBefore,newCellCMBeforeBefore, newCellCMAfter;
	Coordinates3D<double> nCellCMBefore,nCellCMBeforeBefore,nCellCMAfter;

	Coordinates3D<double> oldCellVel,nCellVel, newCellVel, distanceInvariantVec;

	CellG *nCell=0;
	WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();

	std::set<NeighborSurfaceData > *oldCellNeighborsPtr=0;
	std::set<NeighborSurfaceData > *newCellNeighborsPtr=0;

	std::set<NeighborSurfaceData >::iterator sitr;

	set<NeighborSurfaceData> oldCellPixelNeighborSurfaceData;
	set<NeighborSurfaceData>::iterator sitrNSD;
	set<NeighborSurfaceData>::iterator sitrNSDTmp;

	bool printFlag=false;


	// precalculateAfterFlipInstantVelocityData(pt,newCell,oldCell);

	if(oldCell){

		//new code
		oldCellNeighborsPtr = &(neighborTrackerAccessorPtr->get(oldCell->extraAttribPtr)->cellNeighbors);
		oldCellCMAfter  = precalculateCentroid(pt, oldCell, -1,fieldDim, boundaryStrategy);

		if(oldCell->volume>1){
			oldCellCMAfter.x=oldCellCMAfter.x/(oldCell->volume-1) ;
			oldCellCMAfter.y=oldCellCMAfter.y/(oldCell->volume-1) ;
			oldCellCMAfter.z=oldCellCMAfter.z/(oldCell->volume-1) ;
		}else{
			oldCellCMAfter.x=oldCellCMAfter.x/oldCell->volume ;
			oldCellCMAfter.y=oldCellCMAfter.y/oldCell->volume ;
			oldCellCMAfter.z=oldCellCMAfter.z/oldCell->volume ;
		}

		oldCellCMBefore=Coordinates3D<double>(
			oldCell->xCM/(double)oldCell->volume ,
			oldCell->yCM/(double)oldCell->volume ,
			oldCell->zCM/(double)oldCell->volume

			);

		oldCellCMBeforeBefore=Coordinates3D<double>(oldCell->xCOMPrev,oldCell->yCOMPrev,oldCell->zCOMPrev);               

		//new code


	}
	if(newCell){

		//new code
		newCellNeighborsPtr = &(neighborTrackerAccessorPtr->get(newCell->extraAttribPtr)->cellNeighbors);
		newCellCMAfter = precalculateCentroid(pt, newCell, +1,fieldDim, boundaryStrategy);

		newCellCMAfter.x= newCellCMAfter.x/(newCell->volume+1);
		newCellCMAfter.y= newCellCMAfter.y/(newCell->volume+1);
		newCellCMAfter.z= newCellCMAfter.z/(newCell->volume+1);

		newCellCMBefore = Coordinates3D<double>(newCell->xCM/(double)newCell->volume ,newCell->yCM/(double)newCell->volume ,newCell->zCM/(double)newCell->volume);


		newCellCMBeforeBefore=Coordinates3D<double>(newCell->xCOMPrev,newCell->yCOMPrev,newCell->zCOMPrev);               
	}

	//will compute here common surface area of old cell pixel with its all nearest neighbors

	Neighbor neighbor;
	//will compute here common surface area of old cell pixel with its all nearest neighbors    
	for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
		neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
		if(!neighbor.distance){
			//if distance is 0 then the neighbor returned is invalid
			continue;
		}

		nCell = fieldG->get(neighbor.pt);
		if(!nCell) continue;
		sitrNSD=oldCellPixelNeighborSurfaceData.find(NeighborSurfaceData(nCell));
		if(sitrNSD != oldCellPixelNeighborSurfaceData.end()){
			sitrNSD->incrementCommonSurfaceArea(*sitrNSD);
		}else{
			oldCellPixelNeighborSurfaceData.insert(NeighborSurfaceData(nCell,1));
		}            
	}

	//NOTE: There is a double counting issue count energy between old and new cell - as it is written in the paper
	//energy before flip from old cell
	if(oldCell){


		for(sitr=oldCellNeighborsPtr->begin() ; sitr != oldCellNeighborsPtr->end() ; ++sitr){

			nCell= sitr -> neighborAddress;

			if (!nCell) continue; //in case medium is a nieighbor

			commonArea = sitr -> commonSurfaceArea;

			


			//          if(nCell->type==2){
			//             printFlag=true;
			//          }



			//cell velocity data -  difference!

			nCellCMBefore=Coordinates3D<double>(
				nCell->xCM/(double)nCell->volume ,
				nCell->yCM/(double)nCell->volume ,
				nCell->zCM/(double)nCell->volume

				);

			nCellCMBeforeBefore=Coordinates3D<double>(nCell->xCOMPrev,nCell->yCOMPrev,nCell->zCOMPrev);

			oldCellVel = distanceVectorCoordinatesInvariant(oldCellCMBefore ,oldCellCMBeforeBefore,fieldDim);
			nCellVel = distanceVectorCoordinatesInvariant(nCellCMBefore ,nCellCMBeforeBefore,fieldDim);

			//if (pt.x>50 && pt.x<60){		
			//	//cerr<<"oldCellCMBefore="<<oldCellCMBefore<<" oldCellCMBeforeBefore="<<oldCellCMBeforeBefore<<endl;
			//	cerr<<"oldCellVel="<<oldCellVel<<" nCellVel="<<nCellVel<<endl;
			//}
			velocityDiffX = oldCellVel.x-nCellVel.x;
			velocityDiffY = oldCellVel.y-nCellVel.y;
			velocityDiffZ = oldCellVel.z-nCellVel.z;


			distanceInvariantVec=distanceVectorCoordinatesInvariant(oldCellCMBefore ,nCellCMBefore,fieldDim);

			x0=distanceInvariantVec.x;
			y0=distanceInvariantVec.y;
			z0=distanceInvariantVec.z;

			cellDistance=dist(x0,y0,z0);
			//if (nCell->type==5){
			//	cerr<<"old.id="<<oldCell->id<<" nCell->id="<<nCell->id<<endl;
			//	cerr<<"oldCellVel="<<oldCellVel<<" nCellVel ="<<nCellVel <<endl;
			//	cerr<<"nCellCMBefore="<<nCellCMBefore<<" nCellCMBeforeBefore="<<nCellCMBeforeBefore<<endl;
			//}
			//if (pt.x>50 && pt.x<60){		
			//	cerr<<"old.id="<<oldCell->id<<" nCell->id="<<nCell->id<<endl;
			//	cerr<<"commonArea="<<commonArea<<endl;
			//	cerr<<"oldCellVel="<<oldCellVel<<" nCellVel ="<<nCellVel <<endl;
			//	cerr<<"velocityDiffX ="<<velocityDiffX <<" velocityDiffY= "<<velocityDiffY<<" velocityDiffZ="<<velocityDiffZ<<endl;  
			//	cerr<<"cellDistance="<<cellDistance<<endl;
			//	cerr<<" contribution="<<commonArea*(
			//		velocityDiffX*velocityDiffX*sqrt((y0)*(y0)+(z0)*(z0))
			//		// +velocityDiffY*velocityDiffY*sqrt((z0)*(z0)+(x0)*(x0))
			//		// +velocityDiffZ*velocityDiffZ*sqrt((x0)*(x0)+(y0)*(y0))
			//		)
			//		/(cellDistance*cellDistance*cellDistance)<<endl;;

			//}


			if(nCell==newCell){
				energy-=commonArea*(
					velocityDiffX*velocityDiffX*sqrt((y0)*(y0)+(z0)*(z0))
					+velocityDiffY*velocityDiffY*sqrt((z0)*(z0)+(x0)*(x0))
					+velocityDiffZ*velocityDiffZ*sqrt((x0)*(x0)+(y0)*(y0))
					)
					/(cellDistance*cellDistance*cellDistance);
			}else{
				energy-=commonArea*(
					velocityDiffX*velocityDiffX*sqrt((y0)*(y0)+(z0)*(z0))
					+velocityDiffY*velocityDiffY*sqrt((z0)*(z0)+(x0)*(x0))
					+velocityDiffZ*velocityDiffZ*sqrt((x0)*(x0)+(y0)*(y0))
					)
					/(cellDistance*cellDistance*cellDistance);

			}


		}
		//if (pt.x>50 && pt.x<60){		
		//	cerr<<"old before contribution="<<commonArea*(
		//		velocityDiffX*velocityDiffX*sqrt((y0)*(y0)+(z0)*(z0))
		//		// +velocityDiffY*velocityDiffY*sqrt((z0)*(z0)+(x0)*(x0))
		//		// +velocityDiffZ*velocityDiffZ*sqrt((x0)*(x0)+(y0)*(y0))
		//		)
		//		/(cellDistance*cellDistance*cellDistance)<<endl;;
		//	
		//}

	}


	//energy before flip from new cell
	if(newCell){


		for(sitr=newCellNeighborsPtr->begin() ; sitr != newCellNeighborsPtr->end() ; ++sitr){

			nCell= sitr -> neighborAddress;

			if (!nCell) continue; //in case medium is a nieighbor
			///DOUBLE COUNTING PROTECTION *******************************************************************

			if(nCell==oldCell) continue; //to avoid double counting of newCell-oldCell eenrgy


			commonArea = sitr -> commonSurfaceArea;



			//          if(nCell->type==2){
			//             printFlag=true;
			//          }



			//cell velocity data -  difference!

			nCellCMBefore=Coordinates3D<double>(
				nCell->xCM/(double)nCell->volume ,
				nCell->yCM/(double)nCell->volume ,
				nCell->zCM/(double)nCell->volume

				);

			nCellCMBeforeBefore=Coordinates3D<double>(nCell->xCOMPrev,nCell->yCOMPrev,nCell->zCOMPrev);

			newCellVel = distanceVectorCoordinatesInvariant(newCellCMBefore ,newCellCMBeforeBefore,fieldDim);
			nCellVel = distanceVectorCoordinatesInvariant(nCellCMBefore ,nCellCMBeforeBefore,fieldDim);

			velocityDiffX = newCellVel.x-nCellVel.x;
			velocityDiffY = newCellVel.y-nCellVel.y;
			velocityDiffZ = newCellVel.z-nCellVel.z;





			distanceInvariantVec=distanceVectorCoordinatesInvariant(newCellCMBefore ,nCellCMBefore,fieldDim);

			x0=distanceInvariantVec.x;
			y0=distanceInvariantVec.y;
			z0=distanceInvariantVec.z;

			cellDistance=dist(x0,y0,z0);


			energy-=commonArea*(
				velocityDiffX*velocityDiffX*sqrt((y0)*(y0)+(z0)*(z0))
				 +velocityDiffY*velocityDiffY*sqrt((z0)*(z0)+(x0)*(x0))
				 +velocityDiffZ*velocityDiffZ*sqrt((x0)*(x0)+(y0)*(y0))
				)
				/(cellDistance*cellDistance*cellDistance);
			//if (pt.x>50 && pt.x<60){		
			//	cerr<<"new.id="<<newCell->id<<" nCell->id="<<nCell->id<<endl;
			//	cerr<<"commonArea="<<commonArea<<endl;
			//	cerr<<"newCellVel="<<newCellVel<<" nCellVel ="<<nCellVel <<endl;
			//	cerr<<"velocityDiffX ="<<velocityDiffX <<" velocityDiffY= "<<velocityDiffY<<" velocityDiffZ="<<velocityDiffZ<<endl;  
			//	cerr<<"cellDistance="<<cellDistance<<endl;
			//	cerr<<" contribution="<<commonArea*(
			//		velocityDiffX*velocityDiffX*sqrt((y0)*(y0)+(z0)*(z0))
			//		// +velocityDiffY*velocityDiffY*sqrt((z0)*(z0)+(x0)*(x0))
			//		// +velocityDiffZ*velocityDiffZ*sqrt((x0)*(x0)+(y0)*(y0))
			//		)
			//		/(cellDistance*cellDistance*cellDistance)<<endl;;

			//}
			//if (pt.x>50 && pt.x<60){		
			//	cerr<<"new before contribution="<<commonArea*(
			//		velocityDiffX*velocityDiffX*sqrt((y0)*(y0)+(z0)*(z0))
			//		// +velocityDiffY*velocityDiffY*sqrt((z0)*(z0)+(x0)*(x0))
			//		// +velocityDiffZ*velocityDiffZ*sqrt((x0)*(x0)+(y0)*(y0))
			//		)
			//		/(cellDistance*cellDistance*cellDistance)<<endl;;
			//	
			//}

		}

	}


	//energy after flip from old cell
	//NOTE:old cell can only loose neighbors with one exception - when new and old cells do not touch each other before pixel copy then old cell will gain new neighbor (so will newCell).
	//In such a case we still do calculations in the last section which handles new neighbors of newCell.
	if(oldCell){

		for(sitr=oldCellNeighborsPtr->begin() ; sitr != oldCellNeighborsPtr->end() ; ++sitr){
			nCell= sitr -> neighborAddress;
			if (!nCell) continue; //in case medium is a neighbor
			//will need to adjust commonArea for after flip case
			commonArea = sitr -> commonSurfaceArea;
			sitrNSD = oldCellPixelNeighborSurfaceData.find(NeighborSurfaceData(nCell));

			if(sitrNSD != oldCellPixelNeighborSurfaceData.end() ){
				if(sitrNSD->neighborAddress != newCell){ // if neighbor pixel is not a newCell we decrement commonArea by the oldCellPixelNeighborSurface 
					commonArea-=sitrNSD->commonSurfaceArea;
				}
				else{//otherwise we do the following
					sitrNSDTmp=oldCellPixelNeighborSurfaceData.find(NeighborSurfaceData(const_cast<CellG*>(oldCell)));
					commonArea-=sitrNSD->commonSurfaceArea;//we subtract common area of pixel with newCell
					if(sitrNSDTmp != oldCellPixelNeighborSurfaceData.end()){// in case old cell is not
						//on the list of oldPixelNeighbors
						commonArea+=sitrNSDTmp->commonSurfaceArea;//we add common area of pixel with oldCell
					}

				}
			}

			//                if(sitrNSDTmp != oldCellPixelNeighborSurfaceData.end()){
			//                   ;
			//                }else{
			//                   cerr<<"sitrNSDTmp is poiting to end of the set PROBLEM!!! commonArea="<<sitrNSDTmp->commonSurfaceArea<<endl;
			//                   cerr<<"OLD CELL="<<oldCell<<" NEW CELL="<<newCell<<endl;
			//                   for(set<NeighborSurfaceData>::iterator itr=oldCellPixelNeighborSurfaceData.begin();
			//                   itr!=oldCellPixelNeighborSurfaceData.end();
			//                   ++itr
			//                   ){
			//                      cerr<<"neighborAddress="<<itr->neighborAddress<<endl;
			//                      cerr<<"commonSurfaceArea="<<itr->commonSurfaceArea<<endl;
			//                   }
			//                   exit(0);
			//                }


			if(commonArea<0.0){ //just in case
				commonArea=0.0;
				cerr<<"reached below zero old after"<<endl;
			}
			if(nCell!=newCell){



				nCellCMBefore=Coordinates3D<double>(
					nCell->xCM/(double)nCell->volume ,
					nCell->yCM/(double)nCell->volume ,
					nCell->zCM/(double)nCell->volume
					);

				nCellCMBeforeBefore=Coordinates3D<double>(nCell->xCOMPrev,nCell->yCOMPrev,nCell->zCOMPrev);


				oldCellVel = distanceVectorCoordinatesInvariant(oldCellCMAfter ,oldCellCMBefore,fieldDim);
				nCellVel = distanceVectorCoordinatesInvariant(nCellCMBefore ,nCellCMBeforeBefore,fieldDim); //if nCell is not a new cell then its velocity before and after spin flip is the same - so I am using earlier expression 

				velocityDiffX = oldCellVel.x-nCellVel.x;
				velocityDiffY = oldCellVel.y-nCellVel.y;
				velocityDiffZ = oldCellVel.z-nCellVel.z;



				nCellCMAfter=Coordinates3D<double>(
					nCell->xCM/(double)nCell->volume ,
					nCell->yCM/(double)nCell->volume ,
					nCell->zCM/(double)nCell->volume

					);

			}else{            
				oldCellVel = distanceVectorCoordinatesInvariant(oldCellCMAfter ,oldCellCMBefore,fieldDim);
				nCellVel = distanceVectorCoordinatesInvariant(newCellCMAfter ,newCellCMBefore,fieldDim); 



				velocityDiffX = oldCellVel.x-nCellVel.x;
				velocityDiffY = oldCellVel.y-nCellVel.y;
				velocityDiffZ = oldCellVel.z-nCellVel.z;            

				nCellCMAfter = newCellCMAfter;
			}

			distanceInvariantVec=distanceVectorCoordinatesInvariant(oldCellCMAfter ,nCellCMAfter,fieldDim);

			x0=distanceInvariantVec.x;
			y0=distanceInvariantVec.y;
			z0=distanceInvariantVec.z;

			cellDistance=dist(x0,y0,z0);


			//if (pt.x>50 && pt.x<60){		
			//	cerr<<"old.id="<<oldCell->id<<" nCell->id="<<nCell->id<<" newCell->id="<<newCell->id<<endl;
			//	cerr<<"commonArea="<<commonArea<<endl;
			//	cerr<<"oldCellCMAfter="<<oldCellCMAfter<<" oldCellCMBefore="<<oldCellCMBefore<<endl;
			//	cerr<<"nCellCMBefore="<<nCellCMBefore<<" nCellCMBeforeBefore="<<nCellCMBeforeBefore<<endl;
			//	cerr<<"oldCellVel="<<oldCellVel<<" nCellVel ="<<nCellVel <<endl;
			//	cerr<<"velocityDiffX ="<<velocityDiffX <<" velocityDiffY= "<<velocityDiffY<<" velocityDiffZ="<<velocityDiffZ<<endl;  
			//	cerr<<"cellDistance="<<cellDistance<<endl;
			//	cerr<<" contribution="<<commonArea*(
			//		velocityDiffX*velocityDiffX*sqrt((y0)*(y0)+(z0)*(z0))
			//		// +velocityDiffY*velocityDiffY*sqrt((z0)*(z0)+(x0)*(x0))
			//		// +velocityDiffZ*velocityDiffZ*sqrt((x0)*(x0)+(y0)*(y0))
			//		)
			//		/(cellDistance*cellDistance*cellDistance)<<endl;;

			//}


			if (nCell==newCell){
				energy+=commonArea*(
					velocityDiffX*velocityDiffX*sqrt((y0)*(y0)+(z0)*(z0))
					+velocityDiffY*velocityDiffY*sqrt((z0)*(z0)+(x0)*(x0))
					+velocityDiffZ*velocityDiffZ*sqrt((x0)*(x0)+(y0)*(y0))
					)
					/(cellDistance*cellDistance*cellDistance);
			}else{
				energy+=commonArea*(
					velocityDiffX*velocityDiffX*sqrt((y0)*(y0)+(z0)*(z0))
				    +velocityDiffY*velocityDiffY*sqrt((z0)*(z0)+(x0)*(x0))
					+velocityDiffZ*velocityDiffZ*sqrt((x0)*(x0)+(y0)*(y0))
					)
					/(cellDistance*cellDistance*cellDistance);

			}

		}
	}


	//energy after flip from new cell
	if(newCell){

		for( sitr = newCellNeighborsPtr->begin() ; sitr != newCellNeighborsPtr->end() ; ++sitr ){
			nCell= sitr -> neighborAddress;
			if (!nCell) continue; //in case medium is a nieighbor

			///DOUBLE COUNTING PROTECTION *******************************************************************
			if(nCell==oldCell) continue; //to avoid double counting of newCell-oldCell eenrgy
			//will need to adjust commonArea for after flip case
			commonArea = sitr -> commonSurfaceArea;

			sitrNSD = oldCellPixelNeighborSurfaceData.find(NeighborSurfaceData(nCell));
			if(sitrNSD != oldCellPixelNeighborSurfaceData.end() ){
				if(sitrNSD->neighborAddress != oldCell){ // if neighbor is not a oldCell we increment commonArea
					commonArea+=sitrNSD->commonSurfaceArea;
				}
				else{//otherwise we do the following
					sitrNSDTmp=oldCellPixelNeighborSurfaceData.find(NeighborSurfaceData(const_cast<CellG*>(newCell)));
					if(sitrNSDTmp != oldCellPixelNeighborSurfaceData.end()){// in case new cell is not
						//on the list of oldPixelNeighbors
						commonArea-=sitrNSDTmp->commonSurfaceArea;//we subtract common area of pixel with newCell
					}
					commonArea+=sitrNSD->commonSurfaceArea;//we add common area of pixel with oldCell

				}
			}
			if(commonArea<0.0){ //just in case
				commonArea=0.0;
				cerr<<"reached below zero new after"<<endl;
			}


			if(nCell!=oldCell){


				nCellCMBefore=Coordinates3D<double>(
					nCell->xCM/(double)nCell->volume ,
					nCell->yCM/(double)nCell->volume ,
					nCell->zCM/(double)nCell->volume
					);

				nCellCMBeforeBefore=Coordinates3D<double>(nCell->xCOMPrev,nCell->yCOMPrev,nCell->zCOMPrev);

				newCellVel = distanceVectorCoordinatesInvariant(newCellCMAfter ,newCellCMBefore,fieldDim);
				nCellVel = distanceVectorCoordinatesInvariant(nCellCMBefore ,nCellCMBeforeBefore,fieldDim); //if nCell is not an old cell then its velocity before and after spin flip is the same - so I am using earlier expression 

				velocityDiffX = newCellVel.x-nCellVel.x;
				velocityDiffY = newCellVel.y-nCellVel.y;
				velocityDiffZ = newCellVel.z-nCellVel.z;

				nCellCMAfter=Coordinates3D<double>(
					nCell->xCM/(double)nCell->volume ,
					nCell->yCM/(double)nCell->volume ,
					nCell->zCM/(double)nCell->volume
					);         



			}else{
				//this should never get executed
				newCellVel = distanceVectorCoordinatesInvariant(newCellCMAfter ,newCellCMBefore,fieldDim);
				nCellVel = distanceVectorCoordinatesInvariant(oldCellCMAfter ,oldCellCMBefore,fieldDim); 



				velocityDiffX = oldCellVel.x-nCellVel.x;
				velocityDiffY = oldCellVel.y-nCellVel.y;
				velocityDiffZ = oldCellVel.z-nCellVel.z;            

				nCellCMAfter = oldCellCMAfter;
				cerr<<"EXECUTING FORBIDDEN CODE"<<endl;


			}


			distanceInvariantVec=distanceVectorCoordinatesInvariant(newCellCMAfter ,nCellCMAfter,fieldDim);

			x0=distanceInvariantVec.x;
			y0=distanceInvariantVec.y;
			z0=distanceInvariantVec.z;

			cellDistance=dist(x0,y0,z0);




			energy+=commonArea*(
				velocityDiffX*velocityDiffX*sqrt((y0)*(y0)+(z0)*(z0))
				+velocityDiffY*velocityDiffY*sqrt((z0)*(z0)+(x0)*(x0))
				+velocityDiffZ*velocityDiffZ*sqrt((x0)*(x0)+(y0)*(y0))
				)
				/(cellDistance*cellDistance*cellDistance);


		}


	}     


	//exclusive case for new neighbors which may occur after the flip
	// examining neighbors of the oldpixel

	//if (pt.x>50 && pt.x<60){		
	//	cerr<<" before extra lambdaViscosity="<<lambdaViscosity<<" lambdaViscosity*energy="<<lambdaViscosity*energy<<endl;
	//}

	//if(newCell){


	//	for(sitrNSD = oldCellPixelNeighborSurfaceData.begin() ; sitrNSD != oldCellPixelNeighborSurfaceData.end() ;++sitrNSD ){
	//		sitr = newCellNeighborsPtr->find(NeighborSurfaceData(sitrNSD->neighborAddress));
	//		if(sitr==newCellNeighborsPtr->end()){//pixel neighbor does not show up in newCellNeighbors - we have found new neighbor of newCell

	//			nCell = sitrNSD->neighborAddress;
	//			if (!nCell) continue; //in case medium is a nieighbor

	//			if(nCell==newCell || nCell==oldCell) continue; //these cases have been already handled
	//			//if(nCell==newCell) continue; //these cases have been already handled

	//			commonArea = sitrNSD->commonSurfaceArea;
	//			
	//			nCellCMBeforeBefore=Coordinates3D<double>(nCell->xCOMPrev,nCell->yCOMPrev,nCell->zCOMPrev);

	//			nCellCMBefore=Coordinates3D<double>(
	//				nCell->xCM/(double)nCell->volume ,
	//				nCell->yCM/(double)nCell->volume ,
	//				nCell->zCM/(double)nCell->volume
	//				);

	//			newCellVel = distanceVectorCoordinatesInvariant(newCellCMAfter ,newCellCMBefore,fieldDim);
	//			nCellVel = distanceVectorCoordinatesInvariant(nCellCMBefore ,nCellCMBeforeBefore,fieldDim); //new neighbor velocity before and after spin flip is the same - so I am using earlier expression 

	//			velocityDiffX = newCellVel.x-nCellVel.x;
	//			velocityDiffY = newCellVel.y-nCellVel.y;
	//			velocityDiffZ = newCellVel.z-nCellVel.z;

	//			nCellCMAfter=Coordinates3D<double>(
	//				nCell->xCM/(double)nCell->volume ,
	//				nCell->yCM/(double)nCell->volume ,
	//				nCell->zCM/(double)nCell->volume
	//				);         




	//			distanceInvariantVec=distanceVectorCoordinatesInvariant(newCellCMAfter ,nCellCMAfter,fieldDim);

	//			x0=distanceInvariantVec.x;
	//			y0=distanceInvariantVec.y;
	//			z0=distanceInvariantVec.z;

	//			cellDistance=dist(x0,y0,z0);




	//			energy+=commonArea*(
	//				velocityDiffX*velocityDiffX*sqrt((y0)*(y0)+(z0)*(z0))
	//				 +velocityDiffY*velocityDiffY*sqrt((z0)*(z0)+(x0)*(x0))
	//				 +velocityDiffZ*velocityDiffZ*sqrt((x0)*(x0)+(y0)*(y0))
	//				)
	//				/(cellDistance*cellDistance*cellDistance);


	//		}
	//	}
	//}


	//if (pt.x>50 && pt.x<60){		
	//	cerr<<" lambdaViscosity="<<lambdaViscosity<<" lambdaViscosity*energy="<<lambdaViscosity*energy<<endl;
	//}
	return lambdaViscosity*energy;

}




std::string ViscosityPlugin::steerableName(){
	return pluginName;
}

std::string ViscosityPlugin::toString(){
	return pluginName;	
}
