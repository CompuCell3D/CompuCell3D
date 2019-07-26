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
#include "ConnectivityGlobalPlugin.h"




ConnectivityGlobalPlugin::ConnectivityGlobalPlugin() : 
	potts(0), 
	doNotPrecheckConnectivity(false), 
	fast_algorithm(false)
{
	changeEnergyFcnPtr = &ConnectivityGlobalPlugin::changeEnergyLegacy;
}

ConnectivityGlobalPlugin::~ConnectivityGlobalPlugin() {
}

void ConnectivityGlobalPlugin::setConnectivityStrength(CellG * _cell, double _connectivityStrength) {
	if (_cell) {
		connectivityGlobalDataAccessor.get(_cell->extraAttribPtr)->connectivityStrength = _connectivityStrength;
	}
}

double ConnectivityGlobalPlugin::getConnectivityStrength(CellG * _cell) {
	if (_cell) {
		return connectivityGlobalDataAccessor.get(_cell->extraAttribPtr)->connectivityStrength;
	}
	return 0.0;
}


void ConnectivityGlobalPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

	potts = simulator->getPotts();
	potts->getCellFactoryGroupPtr()->registerClass(&connectivityGlobalDataAccessor);
	//potts->registerEnergyFunction(this);
	potts->registerConnectivityConstraint(this); // we give it a special status to run it only when really needed 
	simulator->registerSteerableObject(this);
	update(_xmlData, true);

}

void ConnectivityGlobalPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

	//if (potts->getDisplayUnitsFlag()) {
	//	Unit energyUnit = potts->getEnergyUnit();




	//	CC3DXMLElement * unitsElem = _xmlData->getFirstElement("Units");
	//	if (!unitsElem) { //add Units element
	//		unitsElem = _xmlData->attachElement("Units");
	//	}

	//	if (unitsElem->getFirstElement("PenaltyUnit")) {
	//		unitsElem->getFirstElement("PenaltyUnit")->updateElementValue(energyUnit.toString());
	//	}
	//	else {
	//		CC3DXMLElement * energyElem = unitsElem->attachElement("PenaltyUnit", energyUnit.toString());
	//	}
	//}

	penaltyVec.clear();

	Automaton *automaton = potts->getAutomaton();
	ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)
		set<unsigned char> cellTypesSet;


	map<unsigned char, double> typeIdConnectivityPenaltyMap;

	if (_xmlData->getFirstElement("DoNotPrecheckConnectivity")) {
		doNotPrecheckConnectivity = true;
	}

	if (_xmlData->getFirstElement("FastAlgorithm")) {
		fast_algorithm = true;
		changeEnergyFcnPtr = &ConnectivityGlobalPlugin::changeEnergyFast;
	}


	CC3DXMLElementList penaltyVecXML = _xmlData->getElements("Penalty");

	CC3DXMLElementList connectivityOnVecXML = _xmlData->getElements("ConnectivityOn");

	ASSERT_OR_THROW("You cannot use Penalty and ConnectivityOn tags together. Stick to one convention", !(connectivityOnVecXML.size() && penaltyVecXML.size()));

	// previous ASSERT_OR_THROW will encure that only one of the subsequent for loops be executed 

	for (int i = 0; i < penaltyVecXML.size(); ++i) {
		typeIdConnectivityPenaltyMap.insert(make_pair(automaton->getTypeId(penaltyVecXML[i]->getAttribute("Type")), penaltyVecXML[i]->getDouble()));


		//inserting all the types to the set (duplicate are automatically eleminated) to figure out max value of type Id
		cellTypesSet.insert(automaton->getTypeId(penaltyVecXML[i]->getAttribute("Type")));

	}



	for (int i = 0; i < connectivityOnVecXML.size(); ++i) {
		typeIdConnectivityPenaltyMap.insert(make_pair(automaton->getTypeId(connectivityOnVecXML[i]->getAttribute("Type")), 1.0));
		//inserting all the types to the set (duplicate are automatically eleminated) to figure out max value of type Id
		cellTypesSet.insert(automaton->getTypeId(connectivityOnVecXML[i]->getAttribute("Type")));

	}


	//Now that we know all the types used in the simulation we will find size of the penaltyVec
	vector<unsigned char> cellTypesVector(cellTypesSet.begin(), cellTypesSet.end());//coping set to the vector

	int size = 0;
	if (cellTypesVector.size()) {
		size = *max_element(cellTypesVector.begin(), cellTypesVector.end());
	}

	maxTypeId = size;

	size += 1;//if max element is e.g. 5 then size has to be 6 for an array to be properly allocated



	int index;
	penaltyVec.assign(size, 0.0);
	//inserting connectivity penalty values to penaltyVec;
	for (map<unsigned char, double>::iterator mitr = typeIdConnectivityPenaltyMap.begin(); mitr != typeIdConnectivityPenaltyMap.end(); ++mitr) {
		penaltyVec[mitr->first] = fabs(mitr->second);
	}

	cerr << "size=" << size << endl;
	for (int i = 0; i < size; ++i) {
		cerr << "penaltyVec[" << i << "]=" << penaltyVec[i] << endl;
	}

	//Here I initialize max neighbor index for direct acces to the list of neighbors 
	boundaryStrategy = BoundaryStrategy::getInstance();
	maxNeighborIndex = 0;


	maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);


	cerr << "ConnectivityGlobal maxNeighborIndex=" << maxNeighborIndex << endl;

	
	WatchableField3D<CellG *> *cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
	Dim3D fieldDim = cellFieldG->getDim();

	// max_neighbor_index_local_search is different depending whether we are on hex or cartesian lattice and if this is 2D or 3D simulation
	if (boundaryStrategy->getLatticeType() == HEXAGONAL_LATTICE) { // on hex lattice in 2D and 3D nearest neighbors "completely cover" a given pixel 
		max_neighbor_index_local_search = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);
	}
	else {
		if (fieldDim.x == 1 || fieldDim.y == 1 || fieldDim.z == 1) { //2D simulation 
			max_neighbor_index_local_search = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(2);
		}
		else { //3D
			max_neighbor_index_local_search = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(3);
		}
	}

}



bool ConnectivityGlobalPlugin::checkIfCellIsFragmented(const CellG * cell, Point3D cellPixel) {
	bool cellFragmented = false;

	std::set<Point3D> visitedPixels;

	std::set<Point3D> visitedPixels_0;
	std::set<Point3D> visitedPixels_1;
	std::set<Point3D> visitedPixels_2;

	std::set<Point3D> * vpRef_0 = &visitedPixels_0;
	std::set<Point3D> * vpRef_1 = &visitedPixels_1;
	std::set<Point3D> * vpRef_2 = &visitedPixels_2;


	std::deque<Point3D> filoPointBuffer;


	std::deque<Point3D> filoPointBuffer_0;
	std::deque<Point3D> filoPointBuffer_1;
	std::deque<Point3D> filoPointBuffer_2;

	std::deque<Point3D> * fpbRef_0 = &filoPointBuffer_0;
	std::deque<Point3D> * fpbRef_1 = &filoPointBuffer_1;
	std::deque<Point3D> * fpbRef_2 = &filoPointBuffer_2;


	int visitedPixCounter = 0;
	fpbRef_0->push_back(cellPixel);
	vpRef_1->insert(cellPixel);
	++visitedPixCounter;

	while (!fpbRef_0->empty()) {
		//Point3D currentPoint=filoPointBuffer.front();
		//filoPointBuffer.pop_front();

		Point3D currentPoint = fpbRef_0->front();
		fpbRef_0->pop_front();



		CellG *nCell = 0;
		WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
		Neighbor neighbor;


		for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
			neighbor = boundaryStrategy->getNeighborDirect(currentPoint, nIdx);
			if (!neighbor.distance) {
				//if distance is 0 then the neighbor returned is invalid
				continue;
			}

			nCell = fieldG->get(neighbor.pt);
			if (nCell != cell)
				continue;

			//if(visitedPixels.find(neighbor.pt)!=visitedPixels.end())
			//	continue;//neighbor.pt has already been visited and added to visited points

			if (vpRef_0->find(neighbor.pt) != vpRef_0->end() || vpRef_1->find(neighbor.pt) != vpRef_1->end() || vpRef_2->find(neighbor.pt) != vpRef_2->end())
				continue;//neighbor.pt has already been visited and added to visited points

			//visitedPixels.insert(neighbor.pt);
			//filoPointBuffer.push_back(neighbor.pt);

			fpbRef_1->push_back(neighbor.pt);
			vpRef_2->insert(neighbor.pt);
			//cerr<<" Adding pt="<<neighbor.pt<<" to vps"<<" visitedPixCounter="<<visitedPixCounter<<endl;
			++visitedPixCounter;

		}
		if (fpbRef_0->empty()) {
			//cerr<<"Got empty fpbRef_0"<<endl;
			if (!fpbRef_1->empty()) {

				std::deque<Point3D> * fpbRef_tmp = fpbRef_0;
				fpbRef_0 = fpbRef_1;
				fpbRef_tmp->clear();
				fpbRef_1 = fpbRef_tmp;


				vpRef_0->clear();
				std::set<Point3D> *vpRef_tmp = vpRef_0;
				vpRef_0 = vpRef_1;
				vpRef_1 = vpRef_2;
				vpRef_2 = vpRef_tmp;

			}

		}

	}
	//if(visitedPixels.size()!=(newCell->volume+1)){
	if (visitedPixCounter != (cell->volume)) { //volume of nonfragmented cell calculated using BFT should be the same as actual volume of this cell cell->volume
		cellFragmented = true;
	}

	return cellFragmented;

}

/// <summary>
/// Checks local connectivity in the vicinity of the change pixel
/// </summary>
/// <param name="pt">
/// change pixel - instance of Point3D
/// </param>
/// <param name="cell">
/// ptr to a cell whose connectivity we are tracking 
/// </param>
/// <param name="max_neighbor_index_local_search">
/// point neighborhood -  defines how far from the change pixel we wil be checking if the cell is connected
/// </param>
/// <param name="add_pt_to_bfs"> flag that determines whether change poixel should be included in the BFS or not
/// </param>
/// <returns> boolean flag that tells whether proposed pixel copy woudl keep the cell connected
/// </returns>
bool ConnectivityGlobalPlugin::check_local_connectivity(const Point3D &pt, const CellG *cell, unsigned int max_neighbor_index_local_search, bool add_pt_to_bfs) {

	set<Point3D> cell_pixels_set_reference;
	set<Point3D> cell_pixels_set_variable;
	WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
	CellG * nCell;
	Neighbor neighbor;

	//populating cell_pixel_set with points that are within specified range 
	// and are belonging to a cell whose connectivity we check

	for (unsigned int nIdx = 0; nIdx <= max_neighbor_index_local_search; ++nIdx) {
		neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(pt), nIdx);
		if (!neighbor.distance) {
			//if distance is 0 then the neighbor returned is invalid
			continue;
		}

		nCell = fieldG->get(neighbor.pt);
		if (nCell != cell)
			continue;

		cell_pixels_set_reference.insert(neighbor.pt);
	}

	if (add_pt_to_bfs) {
		//we can only add change point (for newCell only) if it has at least one nearest neighbor

		bool add_pt_to_bfs_ok = false;
		for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
			neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(pt), nIdx);
			if (!neighbor.distance) {
				//if distance is 0 then the neighbor returned is invalid
				continue;
			}

			nCell = fieldG->get(neighbor.pt);
			if (nCell != cell)
				continue;

			add_pt_to_bfs_ok = true;
			break;
		}		
		//if we found nearest neighbor for the pt (when considering newCell only)
		//we add it to the set of newCell pixels
		if (add_pt_to_bfs_ok) {
			cell_pixels_set_reference.insert(pt);
		}
		//otherwise it means that the pt is disconnected from the rest of the newCel and connectivity would be broken
		else {
			return false;
		}
		

		
		//if pt has no nearest neighbors then it means that the newCell will be fragmented and we do not allow this

	}	
	set<Point3D> visited_pixels;
	deque<Point3D> neighbors_deque;
	set<Point3D>::iterator sitr;
	set<Point3D>::iterator sitr_current;

	//start BFS by picking first cell of the set

	cell_pixels_set_variable = cell_pixels_set_reference; //copying reference set

	//sitr_current = cell_pixels_set_reference.begin();
	//Point3D bfs_seed = *sitr_current;

	Point3D bfs_seed = *cell_pixels_set_variable.begin();
	cell_pixels_set_variable.erase(bfs_seed);
	neighbors_deque.push_back(bfs_seed);

	while (!neighbors_deque.empty()) {
		Point3D pt_current = neighbors_deque.front();
		neighbors_deque.pop_front();
		for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
			neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(pt_current), nIdx);


			if (!neighbor.distance) {
				//if distance is 0 then the neighbor returned is invalid
				continue;
			}

			sitr = cell_pixels_set_reference.find(neighbor.pt);
			if (sitr != cell_pixels_set_reference.end()) {
				//neighbors_deque.push_back(neighbor.pt);
				//cell_pixels_set_variable.erase(neighbor.pt);

				if (visited_pixels.find(neighbor.pt) == visited_pixels.end()) {
					neighbors_deque.push_back(neighbor.pt);
					visited_pixels.insert(neighbor.pt);
					//cell_pixels_set.erase(*sitr);
					cell_pixels_set_variable.erase(neighbor.pt);
				}
			}
		}

	}
	
	//at this point BFS should be fininshed and if cell is conneccted the cell_pixels_set should be empty 
	if (cell_pixels_set_variable.empty())
		return true;
	return false;


}

double ConnectivityGlobalPlugin::changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {
	return (this->*changeEnergyFcnPtr)(pt, newCell, oldCell);
}


double ConnectivityGlobalPlugin::changeEnergyFast(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {
	
	//extract local definitions of connectivity strangth and determine which parameters to use - local or by type
	double newCellConnectivityPenalty = 0.0;
	bool newCellByTypeCalculations = false;
	double oldCellConnectivityPenalty = 0.0;
	bool oldCellByTypeCalculations = false;


	if (oldCell) {
		//oldCellConnectivityPenalty = connectivityGlobalDataAccessor.get(oldCell->extraAttribPtr)->connectivityStrength;

		if (oldCell->connectivityOn) {
			oldCellByTypeCalculations = true;
			oldCellConnectivityPenalty = 1.0;
		}
		else if (oldCell->type <= maxTypeId && penaltyVec[oldCell->type] != 0.0) {
			oldCellByTypeCalculations = true;
			oldCellConnectivityPenalty = 1.0;

		}
	}
	if (newCell) {
		//newCellConnectivityPenalty = connectivityGlobalDataAccessor.get(newCell->extraAttribPtr)->connectivityStrength;

		if (newCell->connectivityOn) {
			newCellByTypeCalculations = true;
			newCellConnectivityPenalty = 1.0;
		}
		else if (newCell->type <= maxTypeId && penaltyVec[newCell->type] != 0.0) {
			newCellByTypeCalculations = true;
			newCellConnectivityPenalty = 1.0;

		}
	}

	double penalty = 0.0;
	bool connected = false;
	if (newCell) {
		connected = check_local_connectivity(pt, newCell, max_neighbor_index_local_search, true);
		if (!connected) {
			penalty += newCellConnectivityPenalty;
		}
	}

	if (oldCell) {
		connected = check_local_connectivity(pt, oldCell, max_neighbor_index_local_search, false);
		if (!connected) {
			penalty += oldCellConnectivityPenalty;
		}
	}

	return penalty;

}



//Connectivity constraint based on breadth first traversal of cell pixels
double ConnectivityGlobalPlugin::changeEnergyLegacy(const Point3D &pt, const CellG *newCell, const CellG *oldCell)
{
	//extract local definitions of connectivity strangth and determine which parameters to use - local or by type
	double newCellConnectivityPenalty = 0.0;
	bool newCellByTypeCalculations = false;
	double oldCellConnectivityPenalty = 0.0;
	bool oldCellByTypeCalculations = false;


	if (oldCell) {
		oldCellConnectivityPenalty = connectivityGlobalDataAccessor.get(oldCell->extraAttribPtr)->connectivityStrength;

		if (oldCell->connectivityOn) {
			oldCellByTypeCalculations = true;
			oldCellConnectivityPenalty = 1.0;
		}
		else if (oldCellConnectivityPenalty) { // keeping it for legacy reasons in case people still use setConnectivityStrength API
			oldCellByTypeCalculations = true;
		}

		else if (oldCell->type <= maxTypeId && penaltyVec[oldCell->type] != 0.0) {
			oldCellByTypeCalculations = true;
			oldCellConnectivityPenalty = 1.0;

		}
	}
	if (newCell) {

		if (newCell->connectivityOn) {
			newCellByTypeCalculations = true;
			newCellConnectivityPenalty = 1.0;
		}
		else if (newCellConnectivityPenalty) { // keeping it for legacy reasons in case people still use setConnectivityStrength API
			newCellByTypeCalculations = true;
		}

		else if (newCell->type <= maxTypeId && penaltyVec[newCell->type] != 0.0) {
			newCellByTypeCalculations = true;
			newCellConnectivityPenalty = 1.0;

		}

		//newCellConnectivityPenalty = connectivityGlobalDataAccessor.get(newCell->extraAttribPtr)->connectivityStrength;

		//if (newCell->type <= maxTypeId && penaltyVec[newCell->type] != 0.0) {
		//	newCellByTypeCalculations = true;

		//}

	}


	//first we will check if new or old cells which are subject to connectivity constrains are fragmented. 



	//in case old or new cell is medium fragmented flag is by default set to false


	bool oldCellFragmented = false;
	bool newCellFragmented = false;
	//cerr<<"doNotPrecheckConnectivity="<<doNotPrecheckConnectivity<<endl;
	if (!doNotPrecheckConnectivity) {
		if (oldCell && (oldCellByTypeCalculations || oldCellConnectivityPenalty)) {
			oldCellFragmented = checkIfCellIsFragmented(oldCell, pt);//pt belongs to the old cell before spin flip
			//cerr<<"type="<<(int)oldCell->type<<" oldCellFragmented ="<<oldCellFragmented<<endl;
		}

		if (newCell && (newCellByTypeCalculations || newCellConnectivityPenalty)) {
			Point3D flipNeighborPixel = potts->getFlipNeighbor();
			newCellFragmented = checkIfCellIsFragmented(newCell, flipNeighborPixel);
			//cerr<<"type="<<(int)newCell->type<<" newCellFragmented ="<<newCellFragmented<<endl;

		}

	}


	std::set<Point3D> visitedPixels;

	std::set<Point3D> visitedPixels_0;
	std::set<Point3D> visitedPixels_1;
	std::set<Point3D> visitedPixels_2;

	std::set<Point3D> * vpRef_0 = &visitedPixels_0;
	std::set<Point3D> * vpRef_1 = &visitedPixels_1;
	std::set<Point3D> * vpRef_2 = &visitedPixels_2;


	std::deque<Point3D> filoPointBuffer;


	std::deque<Point3D> filoPointBuffer_0;
	std::deque<Point3D> filoPointBuffer_1;
	std::deque<Point3D> filoPointBuffer_2;

	std::deque<Point3D> * fpbRef_0 = &filoPointBuffer_0;
	std::deque<Point3D> * fpbRef_1 = &filoPointBuffer_1;
	std::deque<Point3D> * fpbRef_2 = &filoPointBuffer_2;


	//assumption: volume has not been updated
	// Remark: the algorithm in this plugin can be further optimized. It is probably not necessary to keep track of all visited points 
	// which should speed up BF traversal .

	double penalty = 0.0;



	if (!newCellFragmented && newCell && (newCellByTypeCalculations || newCellConnectivityPenalty)) {

		double newPenalty = newCellConnectivityPenalty;
		//double newPenalty = 0.0;
		//if (newCellConnectivityPenalty) {
		//	newPenalty = newCellConnectivityPenalty; //locally defined connectivity strength has priority over by-type definition
		//}
		//else {
		//	newPenalty = penaltyVec[newCell->type];
		//}

		//pt becomes newCell's pixel after pixel copy 


		int visitedPixCounter = 0;
		//filoPointBuffer.push_back(pt);
		fpbRef_0->push_back(pt);
		vpRef_1->insert(pt);
		//cerr<<"pt="<<pt<<" visitedPixCounter="<<endl;
		++visitedPixCounter;
		//visitedPixels.insert(pt);

		while (!fpbRef_0->empty()/*!filoPointBuffer.empty()*/) {
			//Point3D currentPoint=filoPointBuffer.front();
			//filoPointBuffer.pop_front();

			Point3D currentPoint = fpbRef_0->front();
			fpbRef_0->pop_front();



			CellG *nCell = 0;
			WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
			Neighbor neighbor;


			for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
				neighbor = boundaryStrategy->getNeighborDirect(currentPoint, nIdx);
				if (!neighbor.distance) {
					//if distance is 0 then the neighbor returned is invalid
					continue;
				}

				nCell = fieldG->get(neighbor.pt);
				if (nCell != newCell)
					continue;

				//if(visitedPixels.find(neighbor.pt)!=visitedPixels.end())
				//	continue;//neighbor.pt has already been visited and added to visited points

				if (vpRef_0->find(neighbor.pt) != vpRef_0->end() || vpRef_1->find(neighbor.pt) != vpRef_1->end() || vpRef_2->find(neighbor.pt) != vpRef_2->end())
					continue;//neighbor.pt has already been visited and added to visited points

				//visitedPixels.insert(neighbor.pt);
				//filoPointBuffer.push_back(neighbor.pt);

				fpbRef_1->push_back(neighbor.pt);
				vpRef_2->insert(neighbor.pt);
				//cerr<<" Adding pt="<<neighbor.pt<<" to vps"<<" visitedPixCounter="<<visitedPixCounter<<endl;
				++visitedPixCounter;

			}
			if (fpbRef_0->empty()) {
				//cerr<<"Got empty fpbRef_0"<<endl;
				if (!fpbRef_1->empty()) {
					//cerr<<"BEFORE ASSIGNMENT"<<endl;
					//cerr<<"fpbRef_0->size()="<<fpbRef_0->size()<<endl;
					//cerr<<"fpbRef_1->size()="<<fpbRef_1->size()<<endl;

					std::deque<Point3D> * fpbRef_tmp = fpbRef_0;
					fpbRef_0 = fpbRef_1;
					//cerr<<"fpbRef_0.size()="<<fpbRef_0->size()<<endl;
					fpbRef_tmp->clear();
					fpbRef_1 = fpbRef_tmp;


					//cerr<<"AFTER ASSIGNMENT"<<endl;
					//cerr<<"fpbRef_0->size()="<<fpbRef_0->size()<<endl;
					//cerr<<"fpbRef_1->size()="<<fpbRef_1->size()<<endl;



					//cerr<<"VP BEFORE ASSIGNMENT"<<endl;
					//cerr<<"vpRef_0->size()="<<vpRef_0->size()<<endl;
					//cerr<<"vpRef_1->size()="<<vpRef_1->size()<<endl;
					//cerr<<"vpRef_2->size()="<<vpRef_2->size()<<endl;
					vpRef_0->clear();
					std::set<Point3D> *vpRef_tmp = vpRef_0;
					vpRef_0 = vpRef_1;
					vpRef_1 = vpRef_2;
					vpRef_2 = vpRef_tmp;

					//cerr<<"VP AFTER ASSIGNMENT"<<endl;
					//cerr<<"vpRef_0->size()="<<vpRef_0->size()<<endl;
					//cerr<<"vpRef_1->size()="<<vpRef_1->size()<<endl;
					//cerr<<"vpRef_2->size()="<<vpRef_2->size()<<endl;

				}

			}

		}
		//if(visitedPixels.size()!=(newCell->volume+1)){
		if (visitedPixCounter != (newCell->volume + 1)) { //we use newCell->volume+1 because we count also the pixel pt that may become part of the new cell after spin flip
			//cerr<<"visitedPixels.size()="<<visitedPixCounter<<" newCell->volume="<<newCell->volume<<endl;
			penalty += newPenalty;
			//cerr<<"new penalty="<<penalty<<endl;
			//exit(0);
		}/*else{
		 cerr<<"visitedPixels.size()="<<visitedPixCounter<<" newCell->volume="<<newCell->volume<<endl;

		 cerr<<"new penalty=0.0"<<endl;
		 exit(0);

		 }*/



	}

	//this reduces chances of holes in the cell but does not eliminate them completely
	if (!oldCellFragmented && !newCell && (oldCellByTypeCalculations || oldCellConnectivityPenalty)) {

		double oldPenalty = oldCellConnectivityPenalty;
		//if(!newCell && oldCell->type<=maxTypeId && penaltyVec[oldCell->type]!=0.0){
		//double oldPenalty = 0.0;
		//if (oldCellConnectivityPenalty) {
		//	oldPenalty = oldCellConnectivityPenalty;
		//}
		//else {
		//	oldPenalty = penaltyVec[oldCell->type];
		//}


		CellG *nCell = 0;
		WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
		Neighbor neighbor;
		bool possibleHole = true;

		for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
			neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt), nIdx);
			if (!neighbor.distance) {
				//if distance is 0 then the neighbor returned is invalid
				continue;
			}

			nCell = fieldG->get(neighbor.pt);
			if (nCell == newCell) {
				possibleHole = false;
			}

		}

		if (possibleHole) {
			penalty += oldPenalty;
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

	if (!oldCellFragmented && oldCell && (oldCellByTypeCalculations || oldCellConnectivityPenalty)) {
		//pick pixel belonging to oldCell - simply pick one of the first nearest neighbors of the pt

		double oldPenalty = oldCellConnectivityPenalty;

		//double oldPenalty = 0.0;
		//if (oldCellConnectivityPenalty) {
		//	oldPenalty = oldCellConnectivityPenalty;
		//}
		//else {
		//	oldPenalty = penaltyVec[oldCell->type];
		//}


		CellG *nCell = 0;
		WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
		Neighbor neighbor;

		int visitedPixCounter = 0;

		for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
			neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt), nIdx);
			if (!neighbor.distance) {
				//if distance is 0 then the neighbor returned is invalid
				continue;
			}

			nCell = fieldG->get(neighbor.pt);
			if (nCell == oldCell) {

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

		while (!fpbRef_0->empty()) {

			Point3D currentPoint = fpbRef_0->front();
			fpbRef_0->pop_front();
			//Point3D currentPoint=filoPointBuffer.front();
			//filoPointBuffer.pop_front();

			for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
				neighbor = boundaryStrategy->getNeighborDirect(currentPoint, nIdx);
				if (!neighbor.distance) {
					//if distance is 0 then the neighbor returned is invalid
					continue;
				}

				nCell = fieldG->get(neighbor.pt);
				if (nCell != oldCell || neighbor.pt == pt)
					continue;

				//if(visitedPixels.find(neighbor.pt)!=visitedPixels.end())
				//	continue;//neighbor.pt has already been visited and added to visited points

				//visitedPixels.insert(neighbor.pt);
				//filoPointBuffer.push_back(neighbor.pt);

				if (vpRef_0->find(neighbor.pt) != vpRef_0->end() || vpRef_1->find(neighbor.pt) != vpRef_1->end() || vpRef_2->find(neighbor.pt) != vpRef_2->end())
					continue;//neighbor.pt has already been visited and added to visited points

				//visitedPixels.insert(neighbor.pt);
				//filoPointBuffer.push_back(neighbor.pt);

				fpbRef_1->push_back(neighbor.pt);
				vpRef_2->insert(neighbor.pt);
				//cerr<<" Adding pt="<<neighbor.pt<<" to vps"<<" visitedPixCounter="<<visitedPixCounter<<endl;
				++visitedPixCounter;
			}
			if (fpbRef_0->empty()) {
				//cerr<<"Got empty fpbRef_0"<<endl;
				if (!fpbRef_1->empty()) {
					//cerr<<"BEFORE ASSIGNMENT"<<endl;
					//cerr<<"fpbRef_0->size()="<<fpbRef_0->size()<<endl;
					//cerr<<"fpbRef_1->size()="<<fpbRef_1->size()<<endl;

					std::deque<Point3D> * fpbRef_tmp = fpbRef_0;
					fpbRef_0 = fpbRef_1;
					//cerr<<"fpbRef_0.size()="<<fpbRef_0->size()<<endl;
					fpbRef_tmp->clear();
					fpbRef_1 = fpbRef_tmp;


					//cerr<<"AFTER ASSIGNMENT"<<endl;
					//cerr<<"fpbRef_0->size()="<<fpbRef_0->size()<<endl;
					//cerr<<"fpbRef_1->size()="<<fpbRef_1->size()<<endl;



					//cerr<<"VP BEFORE ASSIGNMENT"<<endl;
					//cerr<<"vpRef_0->size()="<<vpRef_0->size()<<endl;
					//cerr<<"vpRef_1->size()="<<vpRef_1->size()<<endl;
					//cerr<<"vpRef_2->size()="<<vpRef_2->size()<<endl;
					vpRef_0->clear();
					std::set<Point3D> *vpRef_tmp = vpRef_0;
					vpRef_0 = vpRef_1;
					vpRef_1 = vpRef_2;
					vpRef_2 = vpRef_tmp;

					//cerr<<"VP AFTER ASSIGNMENT"<<endl;
					//cerr<<"vpRef_0->size()="<<vpRef_0->size()<<endl;
					//cerr<<"vpRef_1->size()="<<vpRef_1->size()<<endl;
					//cerr<<"vpRef_2->size()="<<vpRef_2->size()<<endl;

				}

			}
		}


		if (visitedPixCounter != (oldCell->volume - 1)) {// we use (oldCell->volume-1) to acount for the fact the pt may stop belonging to old cell after pixel copy
			//cerr<<"visitedPixCounter="<<visitedPixCounter<<" oldCell->volume="<<oldCell->volume<<endl;
			penalty += oldPenalty;
			//cerr<<"old penalty="<<penalty<<endl;
		}

	}

	return penalty;
}


std::string ConnectivityGlobalPlugin::toString() {
	return "ConnectivityGlobal";
}
std::string ConnectivityGlobalPlugin::steerableName() {
	return toString();
}