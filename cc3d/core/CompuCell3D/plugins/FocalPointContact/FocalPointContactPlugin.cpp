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


#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <CompuCell3D/Potts3D/Potts3D.h>


#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/Automaton/Automaton.h>
#include <PublicUtilities/NumericalUtils.h>
#include <PublicUtilities/ParallelUtilsOpenMP.h>
#include <Utils/Coordinates3D.h>

using namespace CompuCell3D;


#include <BasicUtils/BasicString.h>
#include <BasicUtils/BasicException.h>
#include <iostream>
#include <algorithm>

using namespace std;

#include "FocalPointContactPlugin.h"




FocalPointContactPlugin::FocalPointContactPlugin():pUtils(0),xmlData(0)   {
   lambda=0.0;
	lambdaSpring=0.0;
   offset=0.0;
	targetDistance=0.0;

   

   
}

FocalPointContactPlugin::~FocalPointContactPlugin() {
  
}




void FocalPointContactPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
  potts=simulator->getPotts();
  xmlData=_xmlData;
  simulator->getPotts()->registerEnergyFunctionWithName(this,toString());
  simulator->registerSteerableObject(this);
  
  ///will register FocalPointBoundaryPixelTracker here
  BasicClassAccessorBase * cellFocalPointBoundaryPixelTrackerAccessorPtr=&focalPointBoundaryPixelTrackerAccessor;
   ///************************************************************************************************  
  ///REMARK. HAVE TO USE THE SAME BASIC CLASS ACCESSOR INSTANCE THAT WAS USED TO REGISTER WITH FACTORY
   ///************************************************************************************************  
  potts->getCellFactoryGroupPtr()->registerClass(cellFocalPointBoundaryPixelTrackerAccessorPtr);
  potts->registerCellGChangeWatcher(this);  

    pUtils=simulator->getParallelUtils();
    unsigned int maxNumberOfWorkNodes=pUtils->getMaxNumberOfWorkNodesPotts();        
    returnedJunctionToPoolFlagVec.assign(maxNumberOfWorkNodes,false);
    newJunctionInitiatedFlagVec.assign(maxNumberOfWorkNodes,false);  
    fpbPixDataOldCellBeforeVec.assign(maxNumberOfWorkNodes,FocalPointBoundaryPixelTrackerData());  
    fpbPixDataOldCellAfterVec.assign(maxNumberOfWorkNodes,FocalPointBoundaryPixelTrackerData());  
    fpbPixDataNewCellBeforeVec.assign(maxNumberOfWorkNodes,FocalPointBoundaryPixelTrackerData());  
    fpbPixDataNewCellAfterVec.assign(maxNumberOfWorkNodes,FocalPointBoundaryPixelTrackerData());  
  
}




void FocalPointContactPlugin::extraInit(Simulator *simulator){
	update(xmlData,true);
}

void FocalPointContactPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
	automaton = potts->getAutomaton();
	ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)

	if(_xmlData->getFirstElement("Lambda")){
		lambda=_xmlData->getFirstElement("Lambda")->getDouble();
	}
	if(_xmlData->getFirstElement("LambdaSpring")){
		lambdaSpring=_xmlData->getFirstElement("LambdaSpring")->getDouble();
	}


	if(_xmlData->getFirstElement("MaxNumberOfJunctions")){
		maxNumberOfJunctions=_xmlData->getFirstElement("MaxNumberOfJunctions")->getUInt();
	}

	if(_xmlData->getFirstElement("Offset")){
		offset=_xmlData->getFirstElement("Offset")->getDouble();
	}

	if(_xmlData->getFirstElement("TargetDistance")){
		targetDistance=_xmlData->getFirstElement("TargetDistance")->getDouble();
	}


			//Here I initialize max neighbor index for direct acces to the list of neighbors 
			boundaryStrategy=BoundaryStrategy::getInstance();
			maxNeighborIndex=0;

			maxNeighborIndexJunctionMove=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);
			if(_xmlData->getFirstElement("NeighborOrderJunctionMove")){
				maxNeighborIndexJunctionMove=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(_xmlData->getFirstElement("NeighborOrderJunctionMove")->getUInt());
			}

			if(_xmlData->getFirstElement("Depth")){
				maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromDepth(_xmlData->getFirstElement("Depth")->getDouble());
				//cerr<<"got here will do depth"<<endl;
			}else{
				//cerr<<"got here will do neighbor order"<<endl;
				if(_xmlData->getFirstElement("NeighborOrder")){

					maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(_xmlData->getFirstElement("NeighborOrder")->getUInt());	
				}else{
					maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);

				}

			}

			cerr<<"Contact maxNeighborIndex="<<maxNeighborIndex<<endl;
	
}

double FocalPointContactPlugin::changeEnergySprings(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {
	
	Point3D flipNeighbor=potts->getFlipNeighbor();
	Coordinates3D<double> pixelCopyVector(pt.x-flipNeighbor.x,pt.y-flipNeighbor.y,pt.z-flipNeighbor.z);

	double energy=0.0;

	if(newCell){
		list<FocalPointBoundaryPixelTrackerData > & listRef = focalPointBoundaryPixelTrackerAccessor.get(newCell->extraAttribPtr)->focalJunctionList;
		list<FocalPointBoundaryPixelTrackerData >::iterator litr;

		for(litr=listRef.begin() ; litr!=listRef.end() ; ++litr){
			Coordinates3D<double> junctionSpringVec (litr->pt2.x-litr->pt1.x,litr->pt2.y-litr->pt1.y,litr->pt2.z-litr->pt1.z);

			energy-=pixelCopyVector*junctionSpringVec*lambdaSpring;
		}

	}

	if(oldCell){
		list<FocalPointBoundaryPixelTrackerData > & listRef = focalPointBoundaryPixelTrackerAccessor.get(oldCell->extraAttribPtr)->focalJunctionList;
		list<FocalPointBoundaryPixelTrackerData >::iterator litr;

		for(litr=listRef.begin() ; litr!=listRef.end() ; ++litr){
			Coordinates3D<double> junctionSpringVec (litr->pt2.x-litr->pt1.x,litr->pt2.y-litr->pt1.y,litr->pt2.z-litr->pt1.z);
			if(litr->cell2 && litr->cell2 == newCell){
				energy-=0.0;//this means we have already included such case in the new cell section
			}else{
				energy-=pixelCopyVector*junctionSpringVec*lambdaSpring;
			}
		}
		
	}
	return energy;
}


double FocalPointContactPlugin::potentialFunction(double _lambda,double _offset, double _targetDistance , double _distance){
   return _offset+_lambda*(_distance-_targetDistance)*(_distance-_targetDistance);
}

double FocalPointContactPlugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {
	
   //This plugin will not work properly with periodic boundry conditions. If necessary I can fix it
   
	if (newCell==oldCell) //this may happen if you are trying to assign same cell to one pixel twice 
		return 0.0;
	

   double energy=0.0;
	WatchableField3D<CellG *> *fieldG =(WatchableField3D<CellG *> *) potts->getCellFieldG();

	Neighbor neighbor;
	Neighbor neighborOfNeighbor;
	CellG * nCell;
	CellG * nnCell;

    int currentWorkNodeNumber=pUtils->getCurrentWorkNodeNumber();	
    short &  newJunctionInitiatedFlag = newJunctionInitiatedFlagVec[currentWorkNodeNumber];
    short &  returnedJunctionToPoolFlag = returnedJunctionToPoolFlagVec[currentWorkNodeNumber];    
    FocalPointBoundaryPixelTrackerData & fpbPixDataOldCellBefore=fpbPixDataOldCellBeforeVec[currentWorkNodeNumber];
    FocalPointBoundaryPixelTrackerData & fpbPixDataOldCellAfter=fpbPixDataOldCellAfterVec[currentWorkNodeNumber];
    FocalPointBoundaryPixelTrackerData & fpbPixDataNewCellBefore=fpbPixDataNewCellBeforeVec[currentWorkNodeNumber];
    FocalPointBoundaryPixelTrackerData & fpbPixDataNewCellAfter=fpbPixDataNewCellAfterVec[currentWorkNodeNumber];
   

   
	//check if we need to create new junctions only new cell can initiate junctions
	if(newCell){
      
		if(focalPointBoundaryPixelTrackerAccessor.get(newCell->extraAttribPtr)->focalJunctionList.size()<maxNumberOfJunctions){
         
         for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
            neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
            if(!neighbor.distance){
               //if distance is 0 then the neighbor returned is invalid
               continue;
            }
            nCell=fieldG->get(neighbor.pt);
            if(nCell && nCell!=newCell /*&& nCell->type==newCell->type*/){
               //check if nCell has has a junction with newCell 
               list<FocalPointBoundaryPixelTrackerData > & focalJunctionListRef=focalPointBoundaryPixelTrackerAccessor.get(newCell->extraAttribPtr)->focalJunctionList;
               list<FocalPointBoundaryPixelTrackerData >::iterator litr;
               bool junctionExistsFlag=false;
               for(litr=focalJunctionListRef.begin() ; litr!=focalJunctionListRef.end() ; ++litr){
                  if (litr->cell2==nCell){
                    junctionExistsFlag=true; 
                    break;
                  }
               }
               
 
               
               if(!junctionExistsFlag){
                  set<Point3D> & junctionPointSetRef=focalPointBoundaryPixelTrackerAccessor.get(nCell->extraAttribPtr)->junctionPointsSet;
                  if(junctionPointSetRef.find(neighbor.pt)!=junctionPointSetRef.end())
                     continue; //neighbor.pt is already involved in junction
                  
                  
                  fpbPixDataNewCellAfter.pt1=pt;
                  fpbPixDataNewCellAfter.pt2=neighbor.pt;
                  fpbPixDataNewCellAfter.cell1=const_cast<CellG*>(newCell);
                  fpbPixDataNewCellAfter.cell2=nCell;
						cerr<<"adding junction pt1="<<fpbPixDataNewCellAfter.pt1<<" pt2="<< fpbPixDataNewCellAfter.pt2<<endl;
						cerr<<"fpbPixDataNewCellAfter="<<fpbPixDataNewCellAfter<<endl;


						newJunctionInitiatedFlag=true;
                  break;
               }
            }
               
         }
		}
      if(newJunctionInitiatedFlag){
         cerr<<"fpbPixDataOldCellAfter="<<fpbPixDataNewCellAfter<<endl;
			double distAfter=dist(fpbPixDataNewCellAfter.pt1.x,fpbPixDataNewCellAfter.pt1.y,fpbPixDataNewCellAfter.pt1.z,fpbPixDataNewCellAfter.pt2.x,fpbPixDataNewCellAfter.pt2.y,fpbPixDataNewCellAfter.pt2.z);
         energy+=potentialFunction(lambda,offset,targetDistance,distAfter)-0.0;
			cerr<<"lambda="<<lambda<<" distAfter="<<distAfter<<" targetDistance="<<targetDistance<<" offset="<<offset<<endl;
			cerr<<"ENERGY WITH NEW JUNCTION "<<energy<<endl;
			
         return energy;
      }

	}

   
	if(oldCell){
		
		list<FocalPointBoundaryPixelTrackerData > & focalJunctionListRef=focalPointBoundaryPixelTrackerAccessor.get(oldCell->extraAttribPtr)->focalJunctionList;
		list<FocalPointBoundaryPixelTrackerData >::iterator litr;
		list<FocalPointBoundaryPixelTrackerData >::iterator nextLitr;
		//for (litr=focalJunctionListRef.begin() ; litr!=focalJunctionListRef.end() ; ++litr){
		cerr<<"oldCell.id="<<oldCell->id<<" number of junctions="<<focalJunctionListRef.size()<<endl;
		litr=focalJunctionListRef.begin();
		cerr<<"pt of change="<<pt<<endl;
		while(litr!=focalJunctionListRef.end()){
			if(litr->pt1==pt){//pixel containing junction gets removed - need to move junction to a neighbor
				Point3D newJunctionPoint;
            fpbPixDataOldCellBefore=*litr;
				//will need to find closest point
				double minDistance=100.0;
            bool foundMoveSite=false;
				for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndexJunctionMove ; ++nIdx ){
					neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
					if(!neighbor.distance){
						//if distance is 0 then the neighbor returned is invalid
						continue;
					}
               nCell=fieldG->get(neighbor.pt);
					cerr<<"VISITING "<<neighbor.pt<<" oldCell="<<oldCell<<" nCell="<<nCell<<" minDistance="<<minDistance<<endl;
					if(nCell==oldCell && (minDistance>neighbor.distance)){
                  set<Point3D> & junctionPointSetRef=focalPointBoundaryPixelTrackerAccessor.get(oldCell->extraAttribPtr)->junctionPointsSet;
						for (set<Point3D>::iterator sitr=junctionPointSetRef.begin() ; sitr!=junctionPointSetRef.end() ; ++sitr){
							cerr<<"GOT THIS POINT "<<*sitr<<endl;
						}
						if(junctionPointSetRef.find(neighbor.pt)!=junctionPointSetRef.end()){
							cerr<<"neighbor.pt "<<neighbor.pt<<" is alreadey in the set"<<endl;
                     continue; //neighbor.pt is already involved in junction
                  
						}
						minDistance=neighbor.distance;
						newJunctionPoint=neighbor.pt;
                  foundMoveSite=true;
					}
				}
				//will increment iterators here
				if(!foundMoveSite){//no place to move junction point - move to reservoir
               //fpbPixDataOldCellAfter is untouched
               returnedJunctionToPoolFlag=true;
               break; //only one junction per point is allowed
				}else{
					fpbPixDataOldCellAfter=*litr;
				   fpbPixDataOldCellAfter.pt1=newJunctionPoint;
					break;//only one junction per point is allowed
				}
			}
			++litr;
		}
	
	}   

	if(newCell){
		//in this situation we may move junction point from position before spin flip to pt if pt lies closer to
		//corresponding point in the partner cell
		//otherwise we leave junction point of the newCell untouched
		list<FocalPointBoundaryPixelTrackerData > & focalJunctionListRef=focalPointBoundaryPixelTrackerAccessor.get(newCell->extraAttribPtr)->focalJunctionList;
		list<FocalPointBoundaryPixelTrackerData >::iterator litr;
		list<FocalPointBoundaryPixelTrackerData >::iterator nextLitr;

		cerr<<"newCell.id="<<newCell->id<<" number of junctions="<<focalJunctionListRef.size()<<endl;
		
		//will check if pt is a point that lies closer to junction point of the partner cell
		//first check 

		Point3D newJunctionPoint(0,0,0);


		litr=focalJunctionListRef.begin();
		while(litr!=focalJunctionListRef.end()){
			
			if(dist(pt.x,pt.y,pt.z,litr->pt2.x,litr->pt2.y,litr->pt2.z)<dist(litr->pt1.x,litr->pt1.y,litr->pt1.z,litr->pt2.x,litr->pt2.y,litr->pt2.z)){
				Point3D originalJunctionPoint=litr->pt1;
            fpbPixDataNewCellBefore=*litr;
				fpbPixDataNewCellAfter=fpbPixDataNewCellBefore;
				fpbPixDataNewCellAfter.pt1=pt;
            break; //pt can contain only one junction
			}

			litr++;
		}

	}   

   // //update fpbPixData after manipulations
   // if(fpbPixDataOldCellBefore==fpbPixDataOldCellAfter){
      // //no updates necessary
   // }else{
      
   // }
   
   //calculate energy change
   //first make sure that if both newCell and oldCell fpbPixData got changed than the energy has to be calculated simultaneously for both of them
   if(!(fpbPixDataOldCellBefore==fpbPixDataOldCellAfter) && !(fpbPixDataNewCellBefore==fpbPixDataNewCellAfter)){
      if(returnedJunctionToPoolFlag){
         double distBefore=dist(fpbPixDataOldCellBefore.pt1.x,fpbPixDataOldCellBefore.pt1.y,fpbPixDataOldCellBefore.pt1.z,fpbPixDataOldCellBefore.pt2.x,fpbPixDataOldCellBefore.pt2.y,fpbPixDataOldCellBefore.pt2.z);
          energy+=0.0-potentialFunction(lambda,offset,targetDistance,distBefore);
         
      }else{
         //pixel copy causes shift in junction points in both newCell and oldCell
         double distBefore=dist(fpbPixDataOldCellBefore.pt1.x,fpbPixDataOldCellBefore.pt1.y,fpbPixDataOldCellBefore.pt1.z,fpbPixDataNewCellBefore.pt1.x,fpbPixDataNewCellBefore.pt1.y,fpbPixDataNewCellBefore.pt1.z);      
         double distAfter=dist(fpbPixDataOldCellAfter.pt1.x,fpbPixDataOldCellAfter.pt1.y,fpbPixDataOldCellAfter.pt1.z,fpbPixDataNewCellAfter.pt1.x,fpbPixDataNewCellAfter.pt1.y,fpbPixDataNewCellAfter.pt1.z);            
         
         energy+=potentialFunction(lambda,offset,targetDistance,distAfter)-potentialFunction(lambda,offset,targetDistance,distBefore);
      }
   }else{
         //old cell contributions
         if(fpbPixDataOldCellBefore==fpbPixDataOldCellAfter){
            //no contributions
         }else{
            
            double distBefore=dist(fpbPixDataOldCellBefore.pt1.x,fpbPixDataOldCellBefore.pt1.y,fpbPixDataOldCellBefore.pt1.z,fpbPixDataOldCellBefore.pt2.x,fpbPixDataOldCellBefore.pt2.y,fpbPixDataOldCellBefore.pt2.z);
            double distAfter=dist(fpbPixDataOldCellAfter.pt1.x,fpbPixDataOldCellAfter.pt1.y,fpbPixDataOldCellAfter.pt1.z,fpbPixDataOldCellAfter.pt2.x,fpbPixDataOldCellAfter.pt2.y,fpbPixDataOldCellAfter.pt2.z);

            energy+=potentialFunction(lambda,offset,targetDistance,distAfter)-potentialFunction(lambda,offset,targetDistance,distBefore);
            
         }
         
         //new cell contributions
         if(fpbPixDataNewCellBefore==fpbPixDataNewCellAfter){
            //no contributions
         }else{
            
            double distBefore=dist(fpbPixDataNewCellBefore.pt1.x,fpbPixDataNewCellBefore.pt1.y,fpbPixDataNewCellBefore.pt1.z,fpbPixDataNewCellBefore.pt2.x,fpbPixDataNewCellBefore.pt2.y,fpbPixDataNewCellBefore.pt2.z);
            double distAfter=dist(fpbPixDataNewCellAfter.pt1.x,fpbPixDataNewCellAfter.pt1.y,fpbPixDataNewCellAfter.pt1.z,fpbPixDataNewCellAfter.pt2.x,fpbPixDataNewCellAfter.pt2.y,fpbPixDataNewCellAfter.pt2.z);

            energy+=potentialFunction(lambda,offset,targetDistance,distAfter)-potentialFunction(lambda,offset,targetDistance,distBefore);
            
         }

         
   }
	//cerr<<"pt="<<pt<<" energy="<<energy<<endl;

	energy+=changeEnergySprings(pt,newCell,oldCell);
	return energy;
}

void FocalPointContactPlugin::field3DChange(const Point3D &pt, CellG *newCell,CellG *oldCell){
    
    int currentWorkNodeNumber=pUtils->getCurrentWorkNodeNumber();	
    short &  newJunctionInitiatedFlag = newJunctionInitiatedFlagVec[currentWorkNodeNumber];
    short &  returnedJunctionToPoolFlag = returnedJunctionToPoolFlagVec[currentWorkNodeNumber];    
    FocalPointBoundaryPixelTrackerData & fpbPixDataOldCellBefore=fpbPixDataOldCellBeforeVec[currentWorkNodeNumber];
    FocalPointBoundaryPixelTrackerData & fpbPixDataOldCellAfter=fpbPixDataOldCellAfterVec[currentWorkNodeNumber];
    FocalPointBoundaryPixelTrackerData & fpbPixDataNewCellBefore=fpbPixDataNewCellBeforeVec[currentWorkNodeNumber];
    FocalPointBoundaryPixelTrackerData & fpbPixDataNewCellAfter=fpbPixDataNewCellAfterVec[currentWorkNodeNumber];
    
	cerr<<"pt="<<pt<<endl;
	if(newJunctionInitiatedFlag){
		
		
		FocalPointBoundaryPixelTrackerData fpbPixDataNCellAfter;
		fpbPixDataNCellAfter.pt1=fpbPixDataNewCellAfter.pt2;
      fpbPixDataNCellAfter.pt2=fpbPixDataNewCellAfter.pt1;
      fpbPixDataNCellAfter.cell1=fpbPixDataNewCellAfter.cell2;
      fpbPixDataNCellAfter.cell2=fpbPixDataNewCellAfter.cell1;

		cerr<<"fpbPixDataNCellAfter="<<fpbPixDataNCellAfter<<endl;
		cerr<<"fpbPixDataNewCellAfter="<<fpbPixDataNewCellAfter<<endl;

		list<FocalPointBoundaryPixelTrackerData > & focalJunctionListNewCellRef=focalPointBoundaryPixelTrackerAccessor.get(newCell->extraAttribPtr)->focalJunctionList;
		list<FocalPointBoundaryPixelTrackerData > & focalJunctionListNCellRef=focalPointBoundaryPixelTrackerAccessor.get(fpbPixDataNewCellAfter.cell2->extraAttribPtr)->focalJunctionList;
		focalJunctionListNewCellRef.push_back(fpbPixDataNewCellAfter);
		focalJunctionListNCellRef.push_back(fpbPixDataNCellAfter);

		cerr<<"AFTER ADDING NEW JUNCTION TO CELL "<<newCell->id<<" numberOfJunctions="<<focalJunctionListNewCellRef.size()<<endl;
		cerr<<"AFTER ADDING NEW JUNCTION TO CELL "<<fpbPixDataNewCellAfter.cell2->id<<" numberOfJunctions="<<focalJunctionListNCellRef.size()<<endl;

		cerr<<"*************NEW CELL JUNCTIONS****************"<<endl;
		cerr<<"CELL.ID="<<newCell->id<<endl;
		for(list<FocalPointBoundaryPixelTrackerData >::iterator litr =focalJunctionListNewCellRef.begin() ; litr!=focalJunctionListNewCellRef.end() ; ++litr ){
			cerr<<*litr<<endl;
		}

		cerr<<"*************PARTNER CELL JUNCTIONS****************"<<endl;
		cerr<<" CELL.ID="<<fpbPixDataNewCellAfter.cell2->id<<endl;
		for(list<FocalPointBoundaryPixelTrackerData >::iterator litr =focalJunctionListNCellRef.begin() ; litr!=focalJunctionListNCellRef.end() ; ++litr ){
			cerr<<*litr<<endl;
		}


						double a;
						cerr<<" ADDED JUNCTION Input a number"<<endl;
						cin>>a;

		//updating junction pixel set 
		focalPointBoundaryPixelTrackerAccessor.get(newCell->extraAttribPtr)->junctionPointsSet.insert(fpbPixDataNewCellAfter.pt1);
		focalPointBoundaryPixelTrackerAccessor.get(fpbPixDataNCellAfter.cell1->extraAttribPtr)->junctionPointsSet.insert(fpbPixDataNCellAfter.pt1);

		return;
	}

	//cerr<<" NON INITIALIZE pt="<<pt<<endl;
	if(newCell){

		if(!(fpbPixDataNewCellBefore==fpbPixDataNewCellAfter)){
			
			list<FocalPointBoundaryPixelTrackerData > & focalJunctionListRef=focalPointBoundaryPixelTrackerAccessor.get(newCell->extraAttribPtr)->focalJunctionList;
			list<FocalPointBoundaryPixelTrackerData >::iterator litr;
			for(litr=focalJunctionListRef.begin() ; litr != focalJunctionListRef.end() ; ++litr){
				if(litr->pt1==fpbPixDataNewCellBefore.pt1 && litr->pt2==fpbPixDataNewCellBefore.pt2){

					//updating junction pixel set 
					focalPointBoundaryPixelTrackerAccessor.get(newCell->extraAttribPtr)->junctionPointsSet.erase(litr->pt1);
					focalPointBoundaryPixelTrackerAccessor.get(newCell->extraAttribPtr)->junctionPointsSet.insert(fpbPixDataNewCellAfter.pt1);

					litr->pt1=fpbPixDataNewCellAfter.pt1;										
				}
			}

			list<FocalPointBoundaryPixelTrackerData > & focalJunctionListCell2Ref=focalPointBoundaryPixelTrackerAccessor.get(fpbPixDataNewCellBefore.cell2->extraAttribPtr)->focalJunctionList;
			list<FocalPointBoundaryPixelTrackerData >::iterator litr2;

			for(litr2=focalJunctionListCell2Ref.begin() ; litr2 != focalJunctionListCell2Ref.end() ; ++litr2){
				if(litr2->pt1==fpbPixDataNewCellBefore.pt2){
					
					litr2->pt2=fpbPixDataNewCellAfter.pt1;
					
				}
			}
		}
	}

	if(oldCell){
		
		if(returnedJunctionToPoolFlag){
			cerr<<"RETURNING JUNCTION TO THE POOL "<<endl;
			//removing oldCell fpbPixData
			list<FocalPointBoundaryPixelTrackerData > & focalJunctionListRef=focalPointBoundaryPixelTrackerAccessor.get(oldCell->extraAttribPtr)->focalJunctionList;
			list<FocalPointBoundaryPixelTrackerData >::iterator litr;
			for(litr=focalJunctionListRef.begin() ; litr!=focalJunctionListRef.end() ; ++litr){
				if(litr->pt1==fpbPixDataOldCellBefore.pt1 && litr->cell1==fpbPixDataOldCellBefore.cell1){
					
					CellG *cellN;
					cerr<<"Point being overwritten "<<pt<<" cell Overwriting="<<cellN<<endl;
					if(cellN){
						cerr<<"cellN.id="<<cellN->id<<endl;
					}
					
					//updating junction pixel set 
					focalPointBoundaryPixelTrackerAccessor.get(oldCell->extraAttribPtr)->junctionPointsSet.erase(litr->pt1);
					cerr<<"RETURNED JUNCTION="<<endl;
					cerr<<*litr<<endl;
					cerr<<"BEFORE focalJunctionListRef.size()="<<focalJunctionListRef.size()<<endl;
					focalJunctionListRef.erase(litr);
					cerr<<"AFTER focalJunctionListRef.size()="<<focalJunctionListRef.size()<<endl;
					double a;
					cerr<<"CODE:"<<endl;
					cin>>a;

					break;
				}
			}
			//removing partner fpbPixData 
			list<FocalPointBoundaryPixelTrackerData > & focalJunctionListPartnerCellRef=focalPointBoundaryPixelTrackerAccessor.get(fpbPixDataOldCellBefore.cell2->extraAttribPtr)->focalJunctionList;
			list<FocalPointBoundaryPixelTrackerData >::iterator litrPC;
			for(litrPC=focalJunctionListPartnerCellRef.begin() ; litrPC!=focalJunctionListPartnerCellRef.end() ; ++litrPC){
				if(litrPC->pt2==fpbPixDataOldCellBefore.pt1 && litr->cell2==fpbPixDataOldCellBefore.cell1){

					//updating junction pixel set 
					focalPointBoundaryPixelTrackerAccessor.get(fpbPixDataOldCellBefore.cell2->extraAttribPtr)->junctionPointsSet.erase(litrPC->pt1);

					focalJunctionListPartnerCellRef.erase(litrPC);
					break;
				}
			}
		}else if(!(fpbPixDataOldCellAfter==fpbPixDataOldCellBefore)){
			//updating oldCell
			list<FocalPointBoundaryPixelTrackerData > & focalJunctionListRef=focalPointBoundaryPixelTrackerAccessor.get(oldCell->extraAttribPtr)->focalJunctionList;
			list<FocalPointBoundaryPixelTrackerData >::iterator litr;
			for(litr=focalJunctionListRef.begin() ; litr!=focalJunctionListRef.end() ; ++litr){
				if(litr->pt1==fpbPixDataOldCellBefore.pt1 && litr->cell1==fpbPixDataOldCellBefore.cell1){
					//updating junction pixel set 
					focalPointBoundaryPixelTrackerAccessor.get(oldCell->extraAttribPtr)->junctionPointsSet.erase(litr->pt1);
					focalPointBoundaryPixelTrackerAccessor.get(oldCell->extraAttribPtr)->junctionPointsSet.insert(fpbPixDataOldCellAfter.pt1);

					litr->pt1=fpbPixDataOldCellAfter.pt1;
					break;
				}
			}
			//updating partnerCell
			list<FocalPointBoundaryPixelTrackerData > & focalJunctionListPartnerCellRef=focalPointBoundaryPixelTrackerAccessor.get(fpbPixDataOldCellBefore.cell2->extraAttribPtr)->focalJunctionList;
			list<FocalPointBoundaryPixelTrackerData >::iterator litrPC;
			for(litrPC=focalJunctionListPartnerCellRef.begin() ; litrPC!=focalJunctionListPartnerCellRef.end() ; ++litrPC){
				if(litrPC->pt2==fpbPixDataOldCellBefore.pt1 && litr->cell2==fpbPixDataOldCellBefore.cell1){
					litr->pt2=fpbPixDataOldCellAfter.pt1;
					break;
				}
			}
		}
	
	}
	
	

}

double FocalPointContactPlugin::contactEnergy(const CellG *cell1, const CellG *cell2) {

	return contactEnergyArray[cell1 ? cell1->type : 0][cell2? cell2->type : 0];


}

void FocalPointContactPlugin::setContactEnergy(const string typeName1,const string typeName2,const double energy) {

	char type1 = automaton->getTypeId(typeName1);
	char type2 = automaton->getTypeId(typeName2);

	int index = getIndex(type1, type2);

	contactEnergies_t::iterator it = contactEnergies.find(index);
	ASSERT_OR_THROW(string("Contact energy for ") + typeName1 + " " + typeName2 +
		" already set!", it == contactEnergies.end());

	contactEnergies[index] = energy;
}

int FocalPointContactPlugin::getIndex(const int type1, const int type2) const {
	if (type1 < type2) return ((type1 + 1) | ((type2 + 1) << 16));
	else return ((type2 + 1) | ((type1 + 1) << 16));
}

std::string FocalPointContactPlugin::steerableName(){return "FocalPointContact";}
std::string FocalPointContactPlugin::toString(){return steerableName();}


