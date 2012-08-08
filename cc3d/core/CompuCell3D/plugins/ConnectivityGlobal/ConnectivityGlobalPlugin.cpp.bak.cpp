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
	std::deque<Point3D> filoPointBuffer;
	//assumption: volume has not been updated
	// Remark: the algorithm in this plugin can be further optimized. It is probably not necessary to keep track of all visited points 
	// which should speed up BF traversal .

	double penalty=0.0;

	if(newCell && newCell->type<=maxTypeId){
		//pt becomes newCell's pixel after pixel copy 
		if(penaltyVec[newCell->type]!=0.0){

			filoPointBuffer.push_back(pt);
			visitedPixels.insert(pt);

			while(!filoPointBuffer.empty()){
				Point3D currentPoint=filoPointBuffer.front();
				filoPointBuffer.pop_front();

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

					if(visitedPixels.find(neighbor.pt)!=visitedPixels.end())
						continue;//neighbor.pt has already been visited and added to visited points

					visitedPixels.insert(neighbor.pt);
					filoPointBuffer.push_back(neighbor.pt);

				}

			}
			if(visitedPixels.size()!=(newCell->volume+1)){
				//cerr<<"visitedPixels.size()="<<visitedPixels.size()<<" newCell->volume="<<newCell->volume<<endl;
				penalty+=penaltyVec[newCell->type];
				//cerr<<"new penalty="<<penalty<<endl;
			}

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

	//pt will not belong to oldCell after pixel copy

	if(oldCell && oldCell->type<=maxTypeId){
		//pick pixel belonging to oldCell - simply pick one of the first nearest neighbors of the pt
		if(penaltyVec[oldCell->type]!=0.0 ){
			CellG *nCell=0;
			WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
			Neighbor neighbor;

			for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
				neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
				if(!neighbor.distance){
					//if distance is 0 then the neighbor returned is invalid
					continue;
				}

				nCell = fieldG->get(neighbor.pt);
				if(nCell==oldCell ){

					filoPointBuffer.push_back(neighbor.pt);
					visitedPixels.insert(neighbor.pt);
					// it is essential that you pick only one nearest neighbor of pt and break .If you pick more the connectivity algorithm will not work
					// think about horseshoe shaped cell that is about to break into two pieces
					break;
				}


			}

			while(!filoPointBuffer.empty()){

				Point3D currentPoint=filoPointBuffer.front();
				filoPointBuffer.pop_front();

				for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
					neighbor=boundaryStrategy->getNeighborDirect(currentPoint,nIdx);
					if(!neighbor.distance){
						//if distance is 0 then the neighbor returned is invalid
						continue;
					}

					nCell = fieldG->get(neighbor.pt);
					if(nCell!=oldCell || neighbor.pt==pt)
						continue;

					if(visitedPixels.find(neighbor.pt)!=visitedPixels.end())
						continue;//neighbor.pt has already been visited and added to visited points

					visitedPixels.insert(neighbor.pt);
					filoPointBuffer.push_back(neighbor.pt);

				}

			}


			if(visitedPixels.size()!=(oldCell->volume-1)){
				//cerr<<"visitedPixels.size()="<<visitedPixels.size()<<" oldCell->volume="<<oldCell->volume<<endl;
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