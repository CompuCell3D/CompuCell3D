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
#include <CompuCell3D/Automaton/Automaton.h>
using namespace CompuCell3D;


#include <BasicUtils/BasicString.h>
#include <BasicUtils/BasicException.h>
#include <deque>


#include "ConnectivityGlobalPlugin.h"




ConnectivityGlobalPlugin::ConnectivityGlobalPlugin() : potts(0) 
{
}

ConnectivityGlobalPlugin::~ConnectivityGlobalPlugin() {
}

void ConnectivityGlobalPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

	potts=simulator->getPotts();
	potts->registerEnergyFunction(this);
	simulator->registerSteerableObject(this);
	update(_xmlData,true);

}

void ConnectivityGlobalPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

	penaltyVec.clear();

	Automaton *automaton = potts->getAutomaton();
	ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)
		set<unsigned char> cellTypesSet;

	map<unsigned char,double> typeIdConnectivityPenaltyMap;

	CC3DXMLElementList penaltyVecXML=_xmlData->getElements("Penalty");

	for (int i = 0 ; i<penaltyVecXML.size(); ++i){
		typeIdConnectivityPenaltyMap.insert(make_pair(automaton->getTypeId(penaltyVecXML[i]->getAttribute("Type")),penaltyVecXML[i]->getDouble()));


		//inserting all the types to the set (duplicate are automatically eleminated) to figure out max value of type Id
		cellTypesSet.insert(automaton->getTypeId(penaltyVecXML[i]->getAttribute("Type")));


	}

	//Now that we know all the types used in the simulation we will find size of the penaltyVec
	vector<unsigned char> cellTypesVector(cellTypesSet.begin(),cellTypesSet.end());//coping set to the vector

	int size= * max_element(cellTypesVector.begin(),cellTypesVector.end());
	maxTypeId=size;

	size+=1;//if max element is e.g. 5 then size has to be 6 for an array to be properly allocated

	int index ;
	penaltyVec.assign(size,0.0);
	//inserting connectivity penalty values to penaltyVec;
	for(map<unsigned char , double>::iterator mitr=typeIdConnectivityPenaltyMap.begin() ; mitr!=typeIdConnectivityPenaltyMap.end(); ++mitr){
		penaltyVec[mitr->first]=fabs(mitr->second);
	}

	cerr<<"size="<<size<<endl;
	for(int i = 0 ; i < size ; ++i){
		cerr<<"penaltyVec["<<i<<"]="<<penaltyVec[i]<<endl;
	}

	//Here I initialize max neighbor index for direct acces to the list of neighbors 
	boundaryStrategy=BoundaryStrategy::getInstance();
	maxNeighborIndex=0;


	maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);


	cerr<<"ConnectivityGlobal maxNeighborIndex="<<maxNeighborIndex<<endl;   

}


//Connectivity constraint based on breadth first traversal of cell pixels
double ConnectivityGlobalPlugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) 
{
	std::set<Point3D> visitedPixels;

	std::set<Point3D> visitedPixels_0;
	std::set<Point3D> visitedPixels_1;
	std::set<Point3D> visitedPixels_2;

	std::set<Point3D> * vpRef_0=&visitedPixels_0;
	std::set<Point3D> * vpRef_1=&visitedPixels_1;
	std::set<Point3D> * vpRef_2=&visitedPixels_2;


	std::deque<Point3D> filoPointBuffer;


	std::deque<Point3D> filoPointBuffer_0;
	std::deque<Point3D> filoPointBuffer_1;
	std::deque<Point3D> filoPointBuffer_2;

	std::deque<Point3D> * fpbRef_0=&filoPointBuffer_0;
	std::deque<Point3D> * fpbRef_1=&filoPointBuffer_1;
	std::deque<Point3D> * fpbRef_2=&filoPointBuffer_2;


	//assumption: volume has not been updated
	// Remark: the algorithm in this plugin can be further optimized. It is probably not necessary to keep track of all visited points 
	// which should speed up BF traversal .

	double penalty=0.0;

	if(newCell && newCell->type<=maxTypeId){
		//pt becomes newCell's pixel after pixel copy 
		if(penaltyVec[newCell->type]!=0.0){
			
			int visitedPixCounter=0;
			//filoPointBuffer.push_back(pt);
			fpbRef_0->push_back(pt);
			vpRef_1->insert(pt);
			//cerr<<"pt="<<pt<<" visitedPixCounter="<<endl;
			++visitedPixCounter;
			//visitedPixels.insert(pt);

			while(!fpbRef_0->empty()/*!filoPointBuffer.empty()*/){
				//Point3D currentPoint=filoPointBuffer.front();
				//filoPointBuffer.pop_front();

				Point3D currentPoint=fpbRef_0->front();
				fpbRef_0->pop_front();

				

				CellG *nCell=0;
				WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
				Neighbor neighbor;


				for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
					neighbor=boundaryStrategy->getNeighborDirect(currentPoint,nIdx);
					if(!neighbor.distance){
						//if distance is 0 then the neighbor returned is invalid
						continue;
					}

					nCell = fieldG->get(neighbor.pt);
					if(nCell!=newCell)
						continue;

					//if(visitedPixels.find(neighbor.pt)!=visitedPixels.end())
					//	continue;//neighbor.pt has already been visited and added to visited points

					if(vpRef_0->find(neighbor.pt)!=vpRef_0->end() ||vpRef_1->find(neighbor.pt)!=vpRef_1->end() || vpRef_2->find(neighbor.pt)!=vpRef_2->end())
						continue;//neighbor.pt has already been visited and added to visited points

					//visitedPixels.insert(neighbor.pt);
					//filoPointBuffer.push_back(neighbor.pt);
					
					fpbRef_1->push_back(neighbor.pt);
					vpRef_2->insert(neighbor.pt);
					//cerr<<" Adding pt="<<neighbor.pt<<" to vps"<<" visitedPixCounter="<<visitedPixCounter<<endl;
					++visitedPixCounter;

				}
				if(fpbRef_0->empty()){
					//cerr<<"Got empty fpbRef_0"<<endl;
					if(!fpbRef_1->empty()){
						//cerr<<"BEFORE ASSIGNMENT"<<endl;
						//cerr<<"fpbRef_0->size()="<<fpbRef_0->size()<<endl;
						//cerr<<"fpbRef_1->size()="<<fpbRef_1->size()<<endl;
						
						std::deque<Point3D> * fpbRef_tmp=fpbRef_0;
						fpbRef_0=fpbRef_1;
						//cerr<<"fpbRef_0.size()="<<fpbRef_0->size()<<endl;
						fpbRef_tmp->clear();
						fpbRef_1=fpbRef_tmp;
						

						//cerr<<"AFTER ASSIGNMENT"<<endl;
						//cerr<<"fpbRef_0->size()="<<fpbRef_0->size()<<endl;
						//cerr<<"fpbRef_1->size()="<<fpbRef_1->size()<<endl;



						//cerr<<"VP BEFORE ASSIGNMENT"<<endl;
						//cerr<<"vpRef_0->size()="<<vpRef_0->size()<<endl;
						//cerr<<"vpRef_1->size()="<<vpRef_1->size()<<endl;
						//cerr<<"vpRef_2->size()="<<vpRef_2->size()<<endl;
						vpRef_0->clear();
						std::set<Point3D> *vpRef_tmp=vpRef_0;
						vpRef_0=vpRef_1;
						vpRef_1=vpRef_2;
						vpRef_2=vpRef_tmp;

						//cerr<<"VP AFTER ASSIGNMENT"<<endl;
						//cerr<<"vpRef_0->size()="<<vpRef_0->size()<<endl;
						//cerr<<"vpRef_1->size()="<<vpRef_1->size()<<endl;
						//cerr<<"vpRef_2->size()="<<vpRef_2->size()<<endl;

					}

				}

			}
			//if(visitedPixels.size()!=(newCell->volume+1)){
			if(visitedPixCounter!=(newCell->volume+1)){ //we use newCell->volume+1 because we count also the pixel pt that may become part of the new cell after spin flip
				//cerr<<"visitedPixels.size()="<<visitedPixCounter<<" newCell->volume="<<newCell->volume<<endl;
				penalty+=penaltyVec[newCell->type];
				//cerr<<"new penalty="<<penalty<<endl;
				//exit(0);
			}/*else{
				cerr<<"visitedPixels.size()="<<visitedPixCounter<<" newCell->volume="<<newCell->volume<<endl;
				
				cerr<<"new penalty=0.0"<<endl;
				exit(0);
				
			}*/

		}

	}

	if(!newCell && oldCell->type<=maxTypeId && penaltyVec[oldCell->type]!=0.0){
		
				CellG *nCell=0;
				WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
				Neighbor neighbor;
				bool possibleHole=true;

				for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
					neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
					if(!neighbor.distance){
						//if distance is 0 then the neighbor returned is invalid
						continue;
					}

					nCell = fieldG->get(neighbor.pt);
					if(nCell==newCell){
						possibleHole=false;
					}
						
				}

				if(possibleHole){
					penalty+=penaltyVec[oldCell->type];
				}
	}


	filoPointBuffer.clear();
	visitedPixels.clear();

	fpbRef_0->clear();
	fpbRef_1->clear();

   vpRef_0->clear();
	vpRef_1->clear();
	vpRef_2->clear();


	//pt will not belong to oldCell after pixel copy

	if(oldCell && oldCell->type<=maxTypeId){
		//pick pixel belonging to oldCell - simply pick one of the first nearest neighbors of the pt
		if(penaltyVec[oldCell->type]!=0.0 ){
			CellG *nCell=0;
			WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
			Neighbor neighbor;

			int visitedPixCounter=0;


			for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
				neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
				if(!neighbor.distance){
					//if distance is 0 then the neighbor returned is invalid
					continue;
				}

				nCell = fieldG->get(neighbor.pt);
				if(nCell==oldCell ){

					//filoPointBuffer.push_back(neighbor.pt);
					//visitedPixels.insert(neighbor.pt);

					fpbRef_0->push_back(neighbor.pt);
					vpRef_1->insert(neighbor.pt);
					++visitedPixCounter;

					// it is essential that you pick only one nearest neighbor of pt and break .If you pick more the connectivity algorithm will not work
					// think about horseshoe shaped cell that is about to break into two pieces
					break;
				}


			}

			while(!fpbRef_0->empty()){

				Point3D currentPoint=fpbRef_0->front();
				fpbRef_0->pop_front();
				//Point3D currentPoint=filoPointBuffer.front();
				//filoPointBuffer.pop_front();

				for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
					neighbor=boundaryStrategy->getNeighborDirect(currentPoint,nIdx);
					if(!neighbor.distance){
						//if distance is 0 then the neighbor returned is invalid
						continue;
					}

					nCell = fieldG->get(neighbor.pt);
					if(nCell!=oldCell || neighbor.pt==pt)
						continue;

					//if(visitedPixels.find(neighbor.pt)!=visitedPixels.end())
					//	continue;//neighbor.pt has already been visited and added to visited points

					//visitedPixels.insert(neighbor.pt);
					//filoPointBuffer.push_back(neighbor.pt);

					if(vpRef_0->find(neighbor.pt)!=vpRef_0->end() ||vpRef_1->find(neighbor.pt)!=vpRef_1->end() || vpRef_2->find(neighbor.pt)!=vpRef_2->end())
						continue;//neighbor.pt has already been visited and added to visited points

					//visitedPixels.insert(neighbor.pt);
					//filoPointBuffer.push_back(neighbor.pt);
					
					fpbRef_1->push_back(neighbor.pt);
					vpRef_2->insert(neighbor.pt);
					//cerr<<" Adding pt="<<neighbor.pt<<" to vps"<<" visitedPixCounter="<<visitedPixCounter<<endl;
					++visitedPixCounter;
				}
				if(fpbRef_0->empty()){
					//cerr<<"Got empty fpbRef_0"<<endl;
					if(!fpbRef_1->empty()){
						//cerr<<"BEFORE ASSIGNMENT"<<endl;
						//cerr<<"fpbRef_0->size()="<<fpbRef_0->size()<<endl;
						//cerr<<"fpbRef_1->size()="<<fpbRef_1->size()<<endl;
						
						std::deque<Point3D> * fpbRef_tmp=fpbRef_0;
						fpbRef_0=fpbRef_1;
						//cerr<<"fpbRef_0.size()="<<fpbRef_0->size()<<endl;
						fpbRef_tmp->clear();
						fpbRef_1=fpbRef_tmp;
						

						//cerr<<"AFTER ASSIGNMENT"<<endl;
						//cerr<<"fpbRef_0->size()="<<fpbRef_0->size()<<endl;
						//cerr<<"fpbRef_1->size()="<<fpbRef_1->size()<<endl;



						//cerr<<"VP BEFORE ASSIGNMENT"<<endl;
						//cerr<<"vpRef_0->size()="<<vpRef_0->size()<<endl;
						//cerr<<"vpRef_1->size()="<<vpRef_1->size()<<endl;
						//cerr<<"vpRef_2->size()="<<vpRef_2->size()<<endl;
						vpRef_0->clear();
						std::set<Point3D> *vpRef_tmp=vpRef_0;
						vpRef_0=vpRef_1;
						vpRef_1=vpRef_2;
						vpRef_2=vpRef_tmp;

						//cerr<<"VP AFTER ASSIGNMENT"<<endl;
						//cerr<<"vpRef_0->size()="<<vpRef_0->size()<<endl;
						//cerr<<"vpRef_1->size()="<<vpRef_1->size()<<endl;
						//cerr<<"vpRef_2->size()="<<vpRef_2->size()<<endl;

					}

				}
			}


			if(visitedPixCounter!=(oldCell->volume-1)){// we use (oldCell->volume-1) to acount for the fact the pt may stop belonging to old cell after pixel copy
				//cerr<<"visitedPixCounter="<<visitedPixCounter<<" oldCell->volume="<<oldCell->volume<<endl;
				penalty+=penaltyVec[oldCell->type];
				//cerr<<"old penalty="<<penalty<<endl;
			}
		}
	}

	return penalty;
}


std::string ConnectivityGlobalPlugin::toString(){
	return "ConnectivityGlobal";
}
std::string ConnectivityGlobalPlugin::steerableName(){
	return toString();
}