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

#include "CenterOfMassPlugin.h"

CenterOfMassPlugin::CenterOfMassPlugin():boundaryStrategy(0) {}

CenterOfMassPlugin::~CenterOfMassPlugin() {}

void CenterOfMassPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
	boundaryStrategy=BoundaryStrategy::getInstance();

	cerr<<"\n\n\n  \t\t\t CenterOfMassPlugin::init() - CALLING INIT OF CENTER OF MASS PLUGIN\n\n\n"<<endl;
	potts = simulator->getPotts();
	bool pluginAlreadyRegisteredFlag;
	Plugin *plugin=Simulator::pluginManager.get("VolumeTracker",&pluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded
	if(!pluginAlreadyRegisteredFlag)
		plugin->init(simulator);

	potts->registerCellGChangeWatcher(this);

	potts->getBoundaryXName()=="Periodic" ? boundaryConditionIndicator.x=1 : boundaryConditionIndicator.x=0 ;
	potts->getBoundaryYName()=="Periodic" ? boundaryConditionIndicator.y=1 : boundaryConditionIndicator.y=0;
	potts->getBoundaryZName()=="Periodic" ? boundaryConditionIndicator.z=1 : boundaryConditionIndicator.z=0;

	fieldDim=potts->getCellFieldG()->getDim();

	//determining allowedAreaMin and allowedAreaMax - this seems elaborate but will work for all lattices CC3D supports
	if (boundaryStrategy->getLatticeType()==HEXAGONAL_LATTICE){
		allowedAreaMin.x=0.0;
		allowedAreaMin.y=(fieldDim.z>=3? -sqrt(3.0)/6.0 : 0.0);
		allowedAreaMin.z=0.0;

		allowedAreaMax.x=fieldDim.x+0.5;
		allowedAreaMax.y=fieldDim.y*sqrt(3.0)/2.0+(fieldDim.z>=3? sqrt(3.0)/6.0 : 0.0);
		allowedAreaMax.z=fieldDim.z*sqrt(6.0)/3.0;

	}else{
		allowedAreaMin.x=0.0;
		allowedAreaMin.y=0.0;
		allowedAreaMin.z=0.0;

		allowedAreaMax.x=fieldDim.x;
		allowedAreaMax.y=fieldDim.y;
		allowedAreaMax.z=fieldDim.z;

	}

}

void CompuCell3D::CenterOfMassPlugin::field3DChange(const Point3D &pt, CellG *newCell,CellG * oldCell) {

	if (newCell==oldCell) //this may happen if you are trying to assign same cell to one pixel twice 
		return;

	Coordinates3D<double> ptTrans=boundaryStrategy->calculatePointCoordinates(pt);
	if ( !boundaryConditionIndicator.x && !boundaryConditionIndicator.y && !boundaryConditionIndicator.z ){

		if (oldCell) {
			//temporary code to check if viscosity is working - volume tracker always runs before COM plugin
			if(!potts->checkIfFrozen(oldCell->type)){
				oldCell->xCOMPrev= oldCell->xCM/(oldCell->volume+1);
				oldCell->yCOMPrev= oldCell->yCM/(oldCell->volume+1);
				oldCell->zCOMPrev= oldCell->zCM/(oldCell->volume+1);
			}


			oldCell->xCM -= ptTrans.x;
			oldCell->yCM -= ptTrans.y;
			oldCell->zCM -= ptTrans.z;

			//storing actual center of mass
			if(oldCell->volume){
				oldCell->xCOM = oldCell->xCM /oldCell->volume;
				oldCell->yCOM = oldCell->yCM /oldCell->volume;
				oldCell->zCOM = oldCell->zCM /oldCell->volume;
			}else{
				oldCell->xCOM = 0.0;
				oldCell->yCOM = 0.0;
				oldCell->zCOM = 0.0;
			}

			if(potts->checkIfFrozen(oldCell->type)){
				oldCell->xCOMPrev= oldCell->xCM/(oldCell->volume);
				oldCell->yCOMPrev= oldCell->yCM/(oldCell->volume);
				oldCell->zCOMPrev= oldCell->zCM/(oldCell->volume);
			}
		}

		if (newCell) {
			//temporary code to check if viscosity is working - volume tracker always runs before COM plugin
			if(!potts->checkIfFrozen(newCell->type)){
				if (newCell->volume>1){
					newCell->xCOMPrev= newCell->xCM/(newCell->volume-1);
					newCell->yCOMPrev= newCell->yCM/(newCell->volume-1);
					newCell->zCOMPrev= newCell->zCM/(newCell->volume-1);
				}else{
					newCell->xCOMPrev= newCell->xCM;
					newCell->yCOMPrev= newCell->yCM;
					newCell->zCOMPrev= newCell->zCM;

				}
			}

			newCell->xCM += ptTrans.x;
			newCell->yCM += ptTrans.y;
			newCell->zCM += ptTrans.z;

			//storing actual center of mass
			newCell->xCOM = newCell->xCM /newCell->volume;
			newCell->yCOM = newCell->yCM /newCell->volume;
			newCell->zCOM = newCell->zCM /newCell->volume;

			if(potts->checkIfFrozen(newCell->type)){
				
				newCell->xCOMPrev= newCell->xCM/(newCell->volume);
				newCell->yCOMPrev= newCell->yCM/(newCell->volume);
				newCell->zCOMPrev= newCell->zCM/(newCell->volume);

			}

		}
		return;
	}

	//if there are boundary conditions defined that we have to do some shifts to correctly calculate center of mass
	//This approach will work only for cells whose span is much smaller that lattice dimension in the "periodic "direction
	//e.g. cell that is very long and "wraps lattice" will have miscalculated CM using this algorithm. On the other hand, you do not really expect
	//cells to have dimensions comparable to lattice...

	if (oldCell) {
			//temporary code to check if viscosity is working - volume tracker always runs before COM plugin
			if(!potts->checkIfFrozen(oldCell->type)){
				oldCell->xCOMPrev= oldCell->xCM/(oldCell->volume+1);
				oldCell->yCOMPrev= oldCell->yCM/(oldCell->volume+1);
				oldCell->zCOMPrev= oldCell->zCM/(oldCell->volume+1);
			}


	}

	if (newCell) {
			//temporary code to check if viscosity is working - volume tracker always runs before COM plugin
			if(!potts->checkIfFrozen(newCell->type)){
				if (newCell->volume>1){
					newCell->xCOMPrev= newCell->xCM/(newCell->volume-1);
					newCell->yCOMPrev= newCell->yCM/(newCell->volume-1);
					newCell->zCOMPrev= newCell->zCM/(newCell->volume-1);
				}else{
					newCell->xCOMPrev= newCell->xCM;
					newCell->yCOMPrev= newCell->yCM;
					newCell->zCOMPrev= newCell->zCM;

				}
			}

	}   

	Coordinates3D<double> shiftVec;
	Coordinates3D<double> shiftedPt;
	Coordinates3D<double> distanceVecMin;
	//determines minimum coordinates for the perpendicular lines paccinig through pt
	Coordinates3D<double> distanceVecMax;
	Coordinates3D<double> distanceVecMax_1;
	//determines minimum coordinates for the perpendicular lines paccinig through pt
	Coordinates3D<double> distanceVec; //measures lattice diatances along x,y,z - they can be different for different lattices. The lines have to pass through pt

	distanceVecMin.x=boundaryStrategy->calculatePointCoordinates(Point3D(0,pt.y,pt.z)).x;
	distanceVecMin.y=boundaryStrategy->calculatePointCoordinates(Point3D(pt.x,0,pt.z)).y;
	distanceVecMin.z=boundaryStrategy->calculatePointCoordinates(Point3D(pt.x,pt.y,0)).z;

	distanceVecMax.x=boundaryStrategy->calculatePointCoordinates(Point3D(fieldDim.x,pt.y,pt.z)).x;
	distanceVecMax.y=boundaryStrategy->calculatePointCoordinates(Point3D(pt.x,fieldDim.y,pt.z)).y;
	distanceVecMax.z=boundaryStrategy->calculatePointCoordinates(Point3D(pt.x,pt.y,fieldDim.z)).z;

	distanceVecMax_1.x=boundaryStrategy->calculatePointCoordinates(Point3D(fieldDim.x-1,pt.y,pt.z)).x;
	distanceVecMax_1.y=boundaryStrategy->calculatePointCoordinates(Point3D(pt.x,fieldDim.y-1,pt.z)).y;
	distanceVecMax_1.z=boundaryStrategy->calculatePointCoordinates(Point3D(pt.x,pt.y,fieldDim.z-1)).z;

	distanceVec=distanceVecMax-distanceVecMin;
	//    cerr<<"distanceVec="<<distanceVec<<" distanceVecMin="<<distanceVecMin<<" distanceVecMax="<<distanceVecMax<<endl;

	Coordinates3D<double> fieldDimTrans= boundaryStrategy->calculatePointCoordinates(Point3D(fieldDim.x-1,fieldDim.y-1,fieldDim.z-1));

	double xCM,yCM,zCM; //temporary centroids

	double x,y,z;
	double xo,yo,zo;
	//     cerr<<"CM PLUGIN"<<endl;

	if (oldCell) {
		xo=oldCell->xCM;
		yo=oldCell->yCM;
		zo=oldCell->zCM;

		x=oldCell->xCM-ptTrans.x;
		y=oldCell->yCM-ptTrans.y;
		z=oldCell->zCM-ptTrans.z;

		//calculating shiftVec - to translate CM

		//(oldCell->xCM/(float)(oldCell->volume+1) -pos of CM before th flip - note that volume is updated earlier

		//shift is defined to be zero vector for non-periodic b.c. - everything reduces to naive calculations then   
		shiftVec.x= (oldCell->xCM/(oldCell->volume+1)-((int)fieldDimTrans.x)/2)*boundaryConditionIndicator.x;
		shiftVec.y= (oldCell->yCM/(oldCell->volume+1)-((int)fieldDimTrans.y)/2)*boundaryConditionIndicator.y;
		shiftVec.z= (oldCell->zCM/(oldCell->volume+1)-((int)fieldDimTrans.z)/2)*boundaryConditionIndicator.z;

		//shift CM to approximately center of lattice, new centroids are:
		xCM = oldCell->xCM - shiftVec.x*(oldCell->volume+1);
		yCM = oldCell->yCM - shiftVec.y*(oldCell->volume+1);
		zCM = oldCell->zCM - shiftVec.z*(oldCell->volume+1);
		//Now shift pt
		shiftedPt=ptTrans;
		shiftedPt-=shiftVec;

		//making sure that shifted point is in the lattice
		if(shiftedPt.x < distanceVecMin.x){
			shiftedPt.x += distanceVec.x;
		}else if (shiftedPt.x > distanceVecMax_1.x){
			shiftedPt.x -= distanceVec.x;
		}  

		if(shiftedPt.y < distanceVecMin.y){
			shiftedPt.y += distanceVec.y;
		}else if (shiftedPt.y > distanceVecMax_1.y){
			shiftedPt.y -= distanceVec.y;
		}  

		if(shiftedPt.z < distanceVecMin.z){
			shiftedPt.z += distanceVec.z;
		}else if (shiftedPt.z > distanceVecMax_1.z){
			shiftedPt.z -= distanceVec.z;
		}
		//update shifted centroids
		xCM -= shiftedPt.x;
		yCM -= shiftedPt.y;
		zCM -= shiftedPt.z;

		//shift back centroids
		xCM += shiftVec.x * oldCell->volume;
		yCM += shiftVec.y * oldCell->volume;
		zCM += shiftVec.z * oldCell->volume;

		//Check if CM is in the allowed area
		if( xCM/(float)oldCell->volume < allowedAreaMin.x){
			xCM += distanceVec.x*oldCell->volume;
		}else if ( xCM/(float)oldCell->volume > allowedAreaMax.x){ //will allow to have xCM/vol slightly bigger (by 1) value than max lattice point
			//to avoid rollovers for unsigned int from oldCell->xCM

			//       cerr<<"\t\t\tShifting centroid xCM="<<xCM/(float)oldCell->volume<<endl;
			xCM -= distanceVec.x*oldCell->volume;
			//       cerr<<"\t\t\tShiftedxCM="<<xCM/(float)oldCell->volume<<endl;

		}

		if( yCM/(float)oldCell->volume < allowedAreaMin.y){
			yCM += distanceVec.y*oldCell->volume;
		}else if ( yCM/(float)oldCell->volume > allowedAreaMax.y){
			yCM -= distanceVec.y*oldCell->volume;
		}

		if( zCM/(float)oldCell->volume < allowedAreaMin.z){
			zCM += distanceVec.z*oldCell->volume;
		}else if ( zCM/(float)oldCell->volume > allowedAreaMax.z){
			zCM -= distanceVec.z*oldCell->volume;
		}



		////Check if CM is in the lattice
		//if( xCM/(float)oldCell->volume < distanceVecMin.x){
		//	xCM += distanceVec.x*oldCell->volume;
		//}else if ( xCM/(float)oldCell->volume > distanceVecMax.x){ //will allow to have xCM/vol slightly bigger (by 1) value than max lattice point
		//	//to avoid rollovers for unsigned int from oldCell->xCM

		//	//       cerr<<"\t\t\tShifting centroid xCM="<<xCM/(float)oldCell->volume<<endl;
		//	xCM -= distanceVec.x*oldCell->volume;
		//	//       cerr<<"\t\t\tShiftedxCM="<<xCM/(float)oldCell->volume<<endl;

		//}

		//if( yCM/(float)oldCell->volume < distanceVecMin.y){
		//	yCM += distanceVec.y*oldCell->volume;
		//}else if ( yCM/(float)oldCell->volume > distanceVecMax.y){
		//	yCM -= distanceVec.y*oldCell->volume;
		//}

		//if( zCM/(float)oldCell->volume < distanceVecMin.z){
		//	zCM += distanceVec.z*oldCell->volume;
		//}else if ( zCM/(float)oldCell->volume > distanceVecMax.z){
		//	zCM -= distanceVec.z*oldCell->volume;
		//}

		oldCell->xCM = xCM;
		oldCell->yCM = yCM;
		oldCell->zCM = zCM;

		if(oldCell->volume){
			oldCell->xCOM = oldCell->xCM /oldCell->volume;
			oldCell->yCOM = oldCell->yCM /oldCell->volume;
			oldCell->zCOM = oldCell->zCM /oldCell->volume;
		}else{
			oldCell->xCOM = 0.0;
			oldCell->yCOM = 0.0;
			oldCell->zCOM = 0.0;
		}

		if(potts->checkIfFrozen(oldCell->type)){
			oldCell->xCOMPrev= oldCell->xCM/(oldCell->volume);
			oldCell->yCOMPrev= oldCell->yCM/(oldCell->volume);
			oldCell->zCOMPrev= oldCell->zCM/(oldCell->volume);
		}
		//    cerr<<" oldCell->xCM="<<oldCell->xCM<<" oldCell->yCM="<<oldCell->yCM<<" oldCell->zCM="<<oldCell->zCM<<endl;
	}

	if (newCell) {
		xo=newCell->xCM;
		yo=newCell->yCM;
		zo=newCell->zCM;

		x=newCell->xCM+pt.x;
		y=newCell->yCM+pt.y;
		z=newCell->zCM+pt.z;

		if(newCell->volume==1){
			shiftVec.x=0;
			shiftVec.y=0;
			shiftVec.z=0;
		}else{
			shiftVec.x= (newCell->xCM/(newCell->volume-1)-((int)fieldDimTrans.x)/2)*boundaryConditionIndicator.x;
			shiftVec.y= (newCell->yCM/(newCell->volume-1)-((int)fieldDimTrans.y)/2)*boundaryConditionIndicator.y;
			shiftVec.z= (newCell->zCM/(newCell->volume-1)-((int)fieldDimTrans.z)/2)*boundaryConditionIndicator.z;
		}

		//shift CM to approximately center of lattice , new centroids are:
		xCM = newCell->xCM - shiftVec.x*(newCell->volume-1);
		yCM = newCell->yCM - shiftVec.y*(newCell->volume-1);
		zCM = newCell->zCM - shiftVec.z*(newCell->volume-1);
		//Now shift pt
		shiftedPt=ptTrans;
		shiftedPt-=shiftVec;

		//making sure that shifted point is in the lattice
		if(shiftedPt.x < distanceVecMin.x){
			shiftedPt.x += distanceVec.x;
		}else if (shiftedPt.x > distanceVecMax_1.x){
			//       cerr<<"shifted pt="<<shiftedPt<<endl;
			shiftedPt.x -= distanceVec.x;
		}  

		if(shiftedPt.y < distanceVecMin.y){
			shiftedPt.y += distanceVec.y;
		}else if (shiftedPt.y > distanceVecMax_1.y){
			shiftedPt.y -= distanceVec.y;
		}  

		if(shiftedPt.z < distanceVecMin.z){
			shiftedPt.z += distanceVec.z;
		}else if (shiftedPt.z > distanceVecMax_1.z){
			shiftedPt.z -= distanceVec.z;
		}    

		//update shifted centroids
		xCM += shiftedPt.x;
		yCM += shiftedPt.y;
		zCM += shiftedPt.z;

		//shift back centroids
		xCM += shiftVec.x * newCell->volume;
		yCM += shiftVec.y * newCell->volume;
		zCM += shiftVec.z * newCell->volume;

		//Check if CM is in the lattice
		if( xCM/(float)newCell->volume < allowedAreaMin.x){
			xCM += distanceVec.x*newCell->volume;
		}else if ( xCM/(float)newCell->volume > allowedAreaMax.x){ //will allow to have xCM/vol slightly bigger (by 1) value than max lattice point
			//to avoid rollovers for unsigned int from oldCell->xCM
			xCM -= distanceVec.x*newCell->volume;
		}

		if( yCM/(float)newCell->volume < allowedAreaMin.y){
			yCM += distanceVec.y*newCell->volume;
		}else if ( yCM/(float)newCell->volume > allowedAreaMax.y){
			yCM -= distanceVec.y*newCell->volume;
		}

		if( zCM/(float)newCell->volume < allowedAreaMin.z){
			zCM += distanceVec.z*newCell->volume;
		}else if ( zCM/(float)newCell->volume > allowedAreaMax.z){
			zCM -= distanceVec.z*newCell->volume;
		}


		////Check if CM is in the lattice
		//if( xCM/(float)newCell->volume < distanceVecMin.x){
		//	xCM += distanceVec.x*newCell->volume;
		//}else if ( xCM/(float)newCell->volume > distanceVecMax.x){ //will allow to have xCM/vol slightly bigger (by 1) value than max lattice point
		//	//to avoid rollovers for unsigned int from oldCell->xCM
		//	xCM -= distanceVec.x*newCell->volume;
		//}

		//if( yCM/(float)newCell->volume < distanceVecMin.y){
		//	yCM += distanceVec.y*newCell->volume;
		//}else if ( yCM/(float)newCell->volume > distanceVecMax.y){
		//	yCM -= distanceVec.y*newCell->volume;
		//}

		//if( zCM/(float)newCell->volume < distanceVecMin.z){
		//	zCM += distanceVec.z*newCell->volume;
		//}else if ( zCM/(float)newCell->volume > distanceVecMax.z){
		//	zCM -= distanceVec.z*newCell->volume;
		//}

		newCell->xCM = xCM;
		newCell->yCM = yCM;
		newCell->zCM = zCM;

		//storing actual center of mass
		newCell->xCOM = newCell->xCM /newCell->volume;
		newCell->yCOM = newCell->yCM /newCell->volume;
		newCell->zCOM = newCell->zCM /newCell->volume;

		if(potts->checkIfFrozen(newCell->type)){
			
			newCell->xCOMPrev= newCell->xCM/(newCell->volume);
			newCell->yCOMPrev= newCell->yCM/(newCell->volume);
			newCell->zCOMPrev= newCell->zCM/(newCell->volume);

		}
		//     cerr<<" newCell->xCM="<<newCell->xCM<<" newCell->yCM="<<newCell->yCM<<" newCell->zCM="<<newCell->zCM<<endl;
		//    cerr<<"newCell->xCM="<<newCell->xCM<<" newCell->yCM="<<newCell->yCM<<" newCell->zCM="<<newCell->zCM<<endl;
	}
}

void CenterOfMassPlugin::handleEvent(CC3DEvent & _event){
	if (_event.id!=LATTICE_RESIZE){
		return;
	}
	CC3DEventLatticeResize ev = static_cast<CC3DEventLatticeResize&>(_event);

	Dim3D shiftVec=ev.shiftVec;

    CellInventory &cellInventory = potts->getCellInventory();
    CellInventory::cellInventoryIterator cInvItr;
    CellG * cell;
    
    for(cInvItr=cellInventory.cellInventoryBegin() ; cInvItr !=cellInventory.cellInventoryEnd() ;++cInvItr )
    {
		cell=cInvItr->second;
		//cerr<<"cell->id="<<cell->id<<endl;
		cell->xCOM+=shiftVec.x;
		cell->yCOM+=shiftVec.y;
		cell->zCOM+=shiftVec.z;

		cell->xCOMPrev+=shiftVec.x;
		cell->yCOMPrev+=shiftVec.y;
		cell->zCOMPrev+=shiftVec.z;
		

		cell->xCM+=shiftVec.x*cell->volume;
		cell->yCM+=shiftVec.y*cell->volume;
		cell->zCM+=shiftVec.z*cell->volume;

		
    }


}

//void CenterOfMassPlugin::updateCOMsAfterLatticeShift(Dim3D _shiftVec){
//    CellInventory &cellInventory = potts->getCellInventory();
//    CellInventory::cellInventoryIterator cInvItr;
//    CellG * cell;
//    
//    cerr<<"THIS IS UPDATE COMS"<<endl;
//    for(cInvItr=cellInventory.cellInventoryBegin() ; cInvItr !=cellInventory.cellInventoryEnd() ;++cInvItr )
//    {
//		cell=cInvItr->second;
//		//cerr<<"cell->id="<<cell->id<<endl;
//		cell->xCOM+=_shiftVec.x;
//		cell->yCOM+=_shiftVec.y;
//		cell->zCOM+=_shiftVec.z;
//
//		cell->xCM+=_shiftVec.x*cell->volume;
//		cell->yCM+=_shiftVec.y*cell->volume;
//		cell->zCM+=_shiftVec.z*cell->volume;
//
//    }
//    
//    
//}

std::string CenterOfMassPlugin::toString(){return "CenterOfMass";}
std::string CenterOfMassPlugin::steerableName(){return toString();}

void CenterOfMassPlugin::field3DCheck(const Point3D &pt, CellG *newCell,CellG *oldCell){
	//if no boundary conditions are present
	//    if ( !boundaryConditionIndicator.x && !boundaryConditionIndicator.y && !boundaryConditionIndicator.z ){
	// 
	//       if (oldCell) {
	//          oldCell->xCM -= pt.x;
	//          oldCell->yCM -= pt.y;
	//          oldCell->zCM -= pt.z;
	//       }
	// 
	//       if (newCell) {
	//          newCell->xCM += pt.x;
	//          newCell->yCM += pt.y;
	//          newCell->zCM += pt.z;
	//       }
	//       return;
	//    }
	// 
	// 
	//    Point3D shiftVec;
	//    Point3D shiftedPt;
	//    int xCM,yCM,zCM; //temporary centroids
	// 
	//    int nxCM,nyCM,nzCM; //temporary centroids
	//    int oxCM,oyCM,ozCM; //temporary centroids
	// 
	//    int x,y,z;
	//    int xo,yo,zo;
	// //     cerr<<"CM PLUGIN"<<endl;
	//     
	//   if (oldCell) {
	// 
	//    xo=oldCell->xCM;
	//    yo=oldCell->yCM;
	//    zo=oldCell->zCM;
	// 
	//    x=oldCell->xCM-pt.x;
	//    y=oldCell->yCM-pt.y;
	//    z=oldCell->zCM-pt.z;
	//       
	//     //calculating shiftVec - to translate CM
	// 
	//     //(oldCell->xCM/(float)(oldCell->volume+1) -pos of CM before th flip - note that volume is updated earlier
	// 
	//     //shift is defined to be zero vector for non-periodic b.c. - everything reduces to naive calculations then   
	//     shiftVec.x= (short)((oldCell->xCM/(float)(oldCell->volume+1)-fieldDim.x/2)*boundaryConditionIndicator.x);
	//     shiftVec.y= (short)((oldCell->yCM/(float)(oldCell->volume+1)-fieldDim.y/2)*boundaryConditionIndicator.y);
	//     shiftVec.z= (short)((oldCell->zCM/(float)(oldCell->volume+1)-fieldDim.z/2)*boundaryConditionIndicator.z);
	// 
	// 
	//     //shift CM to approximately center of lattice, new centroids are:
	//     xCM = oldCell->xCM - shiftVec.x*(oldCell->volume+1);
	//     yCM = oldCell->yCM - shiftVec.y*(oldCell->volume+1);
	//     zCM = oldCell->zCM - shiftVec.z*(oldCell->volume+1);
	//     //Now shift pt
	//     shiftedPt=pt;
	//     shiftedPt-=shiftVec;
	//     
	//     //making sure that shifterd point is in the lattice
	//     if(shiftedPt.x < 0){
	//       shiftedPt.x += fieldDim.x;
	//     }else if (shiftedPt.x > fieldDim.x-1){
	//       shiftedPt.x -= fieldDim.x;
	//     }  
	// 
	//     if(shiftedPt.y < 0){
	//       shiftedPt.y += fieldDim.y;
	//     }else if (shiftedPt.y > fieldDim.y-1){
	//       shiftedPt.y -= fieldDim.y;
	//     }  
	// 
	//     if(shiftedPt.z < 0){
	//       shiftedPt.z += fieldDim.z;
	//     }else if (shiftedPt.z > fieldDim.z-1){
	//       shiftedPt.z -= fieldDim.z;
	//     }
	//     //update shifted centroids
	//     xCM -= shiftedPt.x;
	//     yCM -= shiftedPt.y;
	//     zCM -= shiftedPt.z;
	// 
	//     //shift back centroids
	//     xCM += shiftVec.x * oldCell->volume;
	//     yCM += shiftVec.y * oldCell->volume;
	//     zCM += shiftVec.z * oldCell->volume;
	// 
	//     //Check if CM is in the lattice
	//     if( xCM/(float)oldCell->volume < 0){
	//       xCM += fieldDim.x*oldCell->volume;
	//     }else if ( xCM/(float)oldCell->volume > fieldDim.x){ //will allow to have xCM/vol slightly bigger (by 1) value than max lattice point
	//                                                          //to avoid rollovers for unsigned int from oldCell->xCM
	//                                                          
	// //       cerr<<"\t\t\tShifting centroid xCM="<<xCM/(float)oldCell->volume<<endl;
	//       xCM -= fieldDim.x*oldCell->volume;
	// //       cerr<<"\t\t\tShiftedxCM="<<xCM/(float)oldCell->volume<<endl;
	//     }
	// 
	//     if( yCM/(float)oldCell->volume < 0){
	//       yCM += fieldDim.y*oldCell->volume;
	//     }else if ( yCM/(float)oldCell->volume > fieldDim.y){
	//       yCM -= fieldDim.y*oldCell->volume;
	//     }
	// 
	//     if( zCM/(float)oldCell->volume < 0){
	//       zCM += fieldDim.z*oldCell->volume;
	//     }else if ( zCM/(float)oldCell->volume > fieldDim.z){
	//       zCM -= fieldDim.z*oldCell->volume;
	//     }
	//         
	//     oldCell->xCM = xCM;
	//     oldCell->yCM = yCM;
	//     oldCell->zCM = zCM;
	// 
	//     oxCM = xCM;
	//     oyCM = yCM;
	//     ozCM = zCM;
	// 
	//    cerr<<" oxCM="<<oxCM<<" oyCM="<<oyCM<<" ozCM="<<ozCM<<endl;
	//   }
	// 
	//   if (newCell) {
	// 
	//     xo=newCell->xCM;
	//     yo=newCell->yCM;
	//     zo=newCell->zCM;
	// 
	//     x=newCell->xCM+pt.x;
	//     y=newCell->yCM+pt.y;
	//     z=newCell->zCM+pt.z;
	//   
	//     if(newCell->volume==1){
	//       shiftVec.x=0;
	//       shiftVec.y=0;
	//       shiftVec.z=0;
	//       
	//     }else{
	//       shiftVec.x= (short)((newCell->xCM/(float)(newCell->volume-1)-fieldDim.x/2)*boundaryConditionIndicator.x);
	//       shiftVec.y= (short)((newCell->yCM/(float)(newCell->volume-1)-fieldDim.y/2)*boundaryConditionIndicator.y);
	//       shiftVec.z= (short)((newCell->zCM/(float)(newCell->volume-1)-fieldDim.z/2)*boundaryConditionIndicator.z);
	//     }
	//
	//     //shift CM to approximately center of lattice , new centroids are:
	//     xCM = newCell->xCM - shiftVec.x*(newCell->volume-1);
	//     yCM = newCell->yCM - shiftVec.y*(newCell->volume-1);
	//     zCM = newCell->zCM - shiftVec.z*(newCell->volume-1);
	//     //Now shift pt
	//     shiftedPt=pt;
	//     shiftedPt-=shiftVec;
	// 
	//     //making sure that shifted point is in the lattice
	//     if(shiftedPt.x < 0){
	//       shiftedPt.x += fieldDim.x;
	//     }else if (shiftedPt.x > fieldDim.x-1){
	// //       cerr<<"shifted pt="<<shiftedPt<<endl;
	//       shiftedPt.x -= fieldDim.x;
	//     }  
	// 
	//     if(shiftedPt.y < 0){
	//       shiftedPt.y += fieldDim.y;
	//     }else if (shiftedPt.y > fieldDim.y-1){
	//       shiftedPt.y -= fieldDim.y;
	//     }  
	// 
	//     if(shiftedPt.z < 0){
	//       shiftedPt.z += fieldDim.z;
	//     }else if (shiftedPt.z > fieldDim.z-1){
	//       shiftedPt.z -= fieldDim.z;
	//     }    
	// 
	//     //update shifted centroids
	//     xCM += shiftedPt.x;
	//     yCM += shiftedPt.y;
	//     zCM += shiftedPt.z;
	//     
	//     //shift back centroids
	//     xCM += shiftVec.x * newCell->volume;
	//     yCM += shiftVec.y * newCell->volume;
	//     zCM += shiftVec.z * newCell->volume;
	//     
	//     //Check if CM is in the lattice
	//     if( xCM/(float)newCell->volume < 0){
	//       xCM += fieldDim.x*newCell->volume;
	//     }else if ( xCM/(float)newCell->volume > fieldDim.x){ //will allow to have xCM/vol slightly bigger (by 1) value than max lattice point
	//                                                          //to avoid rollovers for unsigned int from oldCell->xCM
	//       xCM -= fieldDim.x*newCell->volume;
	//     }
	// 
	//     if( yCM/(float)newCell->volume < 0){
	//       yCM += fieldDim.y*newCell->volume;
	//     }else if ( yCM/(float)newCell->volume > fieldDim.y){
	//       yCM -= fieldDim.y*newCell->volume;
	//     }
	// 
	//     if( zCM/(float)newCell->volume < 0){
	//       zCM += fieldDim.z*newCell->volume;
	//     }else if ( zCM/(float)newCell->volume > fieldDim.z){
	//       zCM -= fieldDim.z*newCell->volume;
	//     }
	//         
	//     newCell->xCM = xCM;
	//     newCell->yCM = yCM;
	//     newCell->zCM = zCM;
	// 
	//     nxCM = xCM;
	//     nyCM = yCM;
	//     nzCM = zCM;
	// 
	//     cerr<<" nxCM="<<nxCM<<" nyCM="<<nyCM<<" nzCM="<<nzCM<<endl;
	//     
	//   }
}
