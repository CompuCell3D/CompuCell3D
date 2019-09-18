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
* This program is distributed in the hope that it will be useful, but `  *
*      WITHOUT ANY WARRANTY; without even the implied warranty of       *
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    *
*             General Public License for more details.                  *
*                                                                       *
*  You should have received a copy of the GNU General Public License    *
*     along with this program; if not, write to the Free Software       *
*      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.        *
*************************************************************************/


#include "Cell.h"
#include "CellTypeMotilityData.h"
#include "DefaultAcceptanceFunction.h"
#include "AcceptanceFunction.h"
#include "StandardFluctuationAmplitudeFunctions.h"
#include "EnergyFunction.h"
#include "CellGChangeWatcher.h"
#include "Stepper.h"
#include "FixedStepper.h"
#include "AttributeAdder.h"
#include <CompuCell3D/Automaton/Automaton.h>
#include <CompuCell3D/Potts3D/TypeTransition.h>
#include "EnergyFunctionCalculator.h"
#include "EnergyFunctionCalculatorStatistics.h"
#include <CompuCell3D/Simulator.h>
#include <BasicUtils/BasicRandomNumberGenerator.h>
#include <BasicUtils/BasicPluginInfo.h>
#include <PublicUtilities/StringUtils.h>
#include <PublicUtilities/ParallelUtilsOpenMP.h>
#include <deque>
#include <sstream>
#include <algorithm>

#include "Potts3D.h"

using namespace CompuCell3D;
using namespace std;

Potts3D::Potts3D() :
	connectivityConstraint(0),
	cellFieldG(0),
	attrAdder(0),
	acceptanceFunction(&defaultAcceptanceFunction),
	customAcceptanceExpressionDefined(false),
	energy(0),
	depth(1.0),
	displayUnitsFlag(true),
	recentlyCreatedCellId(0),
	recentlyCreatedClusterId(0),
	debugOutputFrequency(10),
	sim(0),
	automaton(0),
	temperature(0.0),
	pUtils(0)

{
	neighbors.assign(100, Point3D());//statically allocated this buffer maybe will come with something better later
	frozenTypeVec.assign(0, 0);
	energyCalculator = new EnergyFunctionCalculator();
	energyCalculator->setPotts(this);
	typeTransition = new TypeTransition();
	metropolisFcnPtr = &Potts3D::metropolisFast;
	cellInventory.setPotts3DPtr(this);
	fluctAmplFcn = new MinFluctuationAmplitudeFunction(this);
}

Potts3D::Potts3D(const Dim3D dim) :
	connectivityConstraint(0),
	cellFieldG(0),
	attrAdder(0),
	acceptanceFunction(&defaultAcceptanceFunction),
	energy(0),
	customAcceptanceExpressionDefined(false),
	depth(1.0),
	displayUnitsFlag(true),
	recentlyCreatedCellId(1),
	recentlyCreatedClusterId(1),
	debugOutputFrequency(10),
	sim(0),
	automaton(0),
	temperature(0.0),
	pUtils(0)

{
	neighbors.assign(100, Point3D());//statically allocated this buffer maybe will come with something better later
	frozenTypeVec.assign(0, 0);

	createCellField(dim);
	energyCalculator = new EnergyFunctionCalculator();
	energyCalculator->setPotts(this);
	typeTransition = new TypeTransition();
	metropolisFcnPtr = &Potts3D::metropolisFast;
	cellInventory.setPotts3DPtr(this);
	fluctAmplFcn = new MinFluctuationAmplitudeFunction(this);

}


Potts3D::~Potts3D() {
	if (cellFieldG) delete cellFieldG;
	if (energyCalculator) delete energyCalculator; energyCalculator = 0;
	if (typeTransition) delete typeTransition; typeTransition = 0;
	//   if (attrAdder) delete attrAdder; attrAdder=0;
	if (fluctAmplFcn) delete fluctAmplFcn;
}

void Potts3D::createEnergyFunction(std::string _energyFunctionType) {
	if (_energyFunctionType == "Statistics") {
		if (energyCalculator) delete energyCalculator; energyCalculator = 0;
		energyCalculator = new EnergyFunctionCalculatorStatistics();
		energyCalculator->setPotts(this);
		//initialize Statistics Output Energy Finction Here
		return;
	}
	else {
		//default is not to reassign energy function calculator
		return;
	}
}

void Potts3D::clean_cell_field(bool reset_cell_inventory) {

	cerr << "cellFieldG=" << cellFieldG << endl;
	if (!cellFieldG) {
		return;
	}

	Point3D pt;
	Dim3D dim_max = cellFieldG->getDim();
	cerr << "dim_max=" << dim_max << endl;

	//cleaning cell field
	for (pt.x = 0; pt.x < dim_max.x; ++pt.x)
		for (pt.y = 0; pt.y < dim_max.y; ++pt.y)
			for (pt.z = 0; pt.z < dim_max.z; ++pt.z) {
				cellFieldG->set(pt, (CellG*)0);
				// this ensures that last pixel deleted will trigger destruction of the cell
				for (unsigned int j = 0; j < steppers.size(); j++) {
					steppers[j]->step();
				}
			}


	if (reset_cell_inventory) {
		recentlyCreatedCellId = 0;
		recentlyCreatedClusterId = 0;
	}

}

LatticeType Potts3D::getLatticeType() {
	return BoundaryStrategy::getInstance()->getLatticeType();
}

void Potts3D::setDepth(double _depth) {
	//this function has to be called after initializing bondary strategy and after creating cellFieldG
	//By default Boundary Strategy will precalculate neighbors up to certain depth (4.0). However if user requests more
	//depth additional calculations will be requested here
	depth = _depth;
	float maxDistance = BoundaryStrategy::getInstance()->getMaxDistance();
	if (maxDistance < depth) {
		//in such situation user requests depth that is greater than default maxDistance 
		BoundaryStrategy::getInstance()->prepareNeighborLists(depth);
	}
	Dim3D dim = cellFieldG->getDim();
	minCoordinates = Point3D(0, 0, 0);
	maxCoordinates = Point3D(dim.x, dim.y, dim.z);

	//will calculate necesary space for neighbor storage
	//this may not work for lattice types other than square

	//   Dim3D dim=cellFieldG->getDim();
	// 
	//   minCoordinates=Point3D(0,0,0);
	//   maxCoordinates=Point3D(dim.x,dim.y,dim.z);
	// 
	// 
	//   Point3D middlePt(dim.x/2,dim.y/2,dim.z/2);
	//   
	//    unsigned int token = 0;
	//    int numNeighbors = 0;
	//    double distance;
	//    Point3D testPt;
	//    token =0;
	//    distance=0;
	//    
	//    
	//       while(true){
	//       testPt = cellFieldG->getNeighbor(middlePt, token, distance);
	//       if (distance > depth) break;
	//    
	//          numNeighbors++;
	//       }
	//    //empty previously allocated container for neighbors
	//    neighbors.clear();
	//    neighbors.assign(numNeighbors+1,Point3D());

	maxNeighborIndex = BoundaryStrategy::getInstance()->getMaxNeighborIndexFromDepth(depth);
	//	cerr << "\t\t\t\t\t setDepth  maxNeighborIndex=" << maxNeighborIndex << endl;
	neighbors.clear();
	neighbors.assign(maxNeighborIndex + 1, Point3D());

}

void Potts3D::setNeighborOrder(unsigned int _neighborOrder) {
	BoundaryStrategy::getInstance()->prepareNeighborListsBasedOnNeighborOrder(_neighborOrder);
	maxNeighborIndex = BoundaryStrategy::getInstance()->getMaxNeighborIndexFromNeighborOrder(_neighborOrder);
	//	cerr << "\t\t\t\t\t setNeighborOrder  maxNeighborIndex=" << maxNeighborIndex << endl;
	Dim3D dim = cellFieldG->getDim();
	minCoordinates = Point3D(0, 0, 0);
	maxCoordinates = Point3D(dim.x, dim.y, dim.z);

	neighbors.clear();
	neighbors.assign(maxNeighborIndex + 1, Point3D());

}


void Potts3D::createCellField(const Dim3D dim) {

	ASSERT_OR_THROW("createCellField() cell field G already created!", !cellFieldG);
	cellFieldG = new WatchableField3D<CellG *>(dim, 0); //added

}

void Potts3D::resizeCellField(const Dim3D dim, Dim3D shiftVec) {

	Dim3D currentDim = cellFieldG->getDim();
	cellFieldG->resizeAndShift(dim, shiftVec);
}


void Potts3D::registerAttributeAdder(AttributeAdder * _attrAdder) {
	attrAdder = _attrAdder;
}

void Potts3D::registerAutomaton(Automaton* autom) { automaton = autom; }
Automaton* Potts3D::getAutomaton() { return automaton; }

void Potts3D::registerEnergyFunction(EnergyFunction *function) {

	energyCalculator->registerEnergyFunctionWithName(function, function->toString());
	//    sim->registerSteerableObject(function);

}

void Potts3D::registerEnergyFunctionWithName(EnergyFunction *_function, std::string _functionName) {

	energyCalculator->registerEnergyFunctionWithName(_function, _functionName);
	//   sim->registerSteerableObject(_function);

}


void Potts3D::unregisterEnergyFunction(std::string _functionName) {

	energyCalculator->unregisterEnergyFunction(_functionName);
	//    sim->unregisterSteerableObject(_functionName);
	return;

}

double Potts3D::getEnergy() { return energy; }

void Potts3D::registerConnectivityConstraint(EnergyFunction * _connectivityConstraint) {
	connectivityConstraint = _connectivityConstraint;
}

EnergyFunction * Potts3D::getConnectivityConstraint() { return connectivityConstraint; }

bool Potts3D::localConectivityAlgorithm(Point3D *changePixel, Point3D *flipNeighbor)
{

	return (this->*localConPtr)(changePixel, flipNeighbor);

	//change goes from flip to change

	/*
	bool localConected = true;


	CellG *changeCell = cellFieldG->get(changePixel);

	CellG *flipCell = cellFieldG->get(flipNeighbor);

	CellG *nCell = 0;

	Neighbor neighbor;



	// check conectivity

	for (unsigned int nIdx = 0; nIdx <= 2; ++nIdx)
	{
		neighbor = boundaryStrategy->getNeighborDirect(
			const_cast<Point3D&>(changePixel), nIdx);
		if (!neighbor.distance) {
			//if distance is 0 then the neighbor returned is invalid
			continue;
		}
		nCell = cellFieldG->get(neighbor.pt);

		if (!nCell)//if it's  medium we don't care
		{
			continue;
		}


		if (nCell == changeCell || nCell == flipCell) //since we've already know the source-target cells we don't need to check for others
		{

			//north => changePixel.y+1 == pt.y; x=x
			//south => changePixel.y-1 == pt.y; x=x

			//east => changePixel.x+1 == pt.x; y=y
			//west => changePixel.x-1 == pt.x; y=y
			if (changePixel.x == neighbor.pt.x)//north-south case
			{


				//might have issues with borders
				Point3D right_pt = neighbor.pt;
				right_pt.x += 1;
				Point3D left_pt = neighbor.pt;
				left_pt.x -= 1;


				if (!(nCell == cellFieldG->get(right_pt) || nCell == cellFieldG->get(left_pt)))
				{
					return localConected = false;
				}


			}
			else if (changePixel.y == neighbor.pt.y)//east-west case
			{
				//might have issues with borders
				Point3D up_pt = neighbor.pt;
				up_pt.y += 1;
				Point3D down_pt = neighbor.pt;
				down_pt.y -= 1;

				if (!(nCell == cellFieldG->get(up_pt) || nCell == cellFieldG->get(down_pt)))
				{
					return localConected = false;
				}


			}


			//corner cases
			else if (changePixel.y+1 == neighbor.pt.y)
			{
				if (changePixel.x + 1 == neighbor.pt.x)//upper right corner
				{
					Point3D left_pt = neighbor.pt;
					left_pt.x -= 1;
					Point3D down_pt = neighbor.pt;
					down_pt.y -= 1;
					if (!(nCell == cellFieldG->get(left_pt) || nCell == cellFieldG->get(down_pt)))
					{
						return localConected = false;
					}
				}
				else if (changePixel.x - 1 == neighbor.pt.x)//upper left corner
				{
					Point3D right_pt = neighbor.pt;
					right_pt.x -= 1;
					Point3D down_pt = neighbor.pt;
					down_pt.y -= 1;
					if (!(nCell == cellFieldG->get(right_pt) || nCell == cellFieldG->get(down_pt)))
					{
						return localConected = false;
					}
				}

			}

			else if (changePixel.y - 1 == neighbor.pt.y)
			{
				if (changePixel.x + 1 == neighbor.pt.x)//lower right corner
				{
					Point3D left_pt = neighbor.pt;
					left_pt.x -= 1;
					Point3D up_pt = neighbor.pt;
					up_pt.y += 1;
					if (!(nCell == cellFieldG->get(left_pt) || nCell == cellFieldG->get(up_pt)))
					{
						return localConected = false;
					}
				}
				else if (changePixel.x - 1 == neighbor.pt.x)//lower left corner
				{
					Point3D right_pt = neighbor.pt;
					right_pt.x -= 1;
					Point3D up_pt = neighbor.pt;
					up_pt.y += 1;
					if (!(nCell == cellFieldG->get(right_pt) || nCell == cellFieldG->get(up_pt)))
					{
						return localConected = false;
					}
				}

			}



		}
	}

	return localConected;*/
}



bool Potts3D::localConect2D(Point3D changePixel, Point3D flipNeighbor)
{

	/*
	* Function to check the local conectivity of a cell in the flip pixel.
	* Based on https://doi.org/10.1016/j.cpc.2016.07.030
	* Durand, Marc, and Etienne Guesnet. "An efficient Cellular Potts Model
	* algorithm that forbids cell fragmentation." Computer Physics
	* Communications 208 (2016): 54-63.
	*
	* Change pixel is the pixel being changed. The value being copied to change
	* pixel in the one of flip neighbor.
	*
	* The algorithm only cares about the imediate vicinity (Moore neighb, order II,
	* all 8 - 2d -  touching pixels) of changePixel.
	*
	* The idea is to get all of the pixels in the neighborhood first. I do this
	* because I have to check some neighbors of the neighbor pixel. I won't cicle
	* through all of the neighbors' neighbors, just check a few. But when near the
	* border I could be picking an invalid pixel, get neighbor direct takes care
	* of checking valied pixels.
	*
	* However, I still will need to figure out the periodic boundary case. How to go to the
	* other side of the lattice eficiently? Better yet, with some method that has been written?
	*
	* Is this method valid for flip distances >1?
	*
	*/

	bool localConected = true;


	CellG *changeCell = cellFieldG->get(changePixel);

	if (!changeCell)return localConected;//if it is medium we don't care

	CellG *flipCell = cellFieldG->get(flipNeighbor);

	CellG *nCell = 0;

	Neighbor neighbor;


	//finalMooreNeighborIndex
	unsigned int fMooreNeighIdx = BoundaryStrategy::getInstance()->getMaxNeighborIndexFromNeighborOrder(2);
	unsigned int fNeumanNeighIdx = BoundaryStrategy::getInstance()->getMaxNeighborIndexFromNeighborOrder(1);



	//what if I make a list of pixels that are in the order 2, then when creating
	// the left/right/up/down pixels I can check that it is in the list.
	//this way I am already using method to check boundaries, hex, etc.
	//But the algorithm will be different for hex! So I need to add a function for that later

	// neighbors

	std::vector<Point3D> search_domain;

	int neig_px_same_cell = 0; //neighboring px to changePixel that are of same cell

	search_domain.clear();

	for (unsigned int nIdx = 0; nIdx <= fNeumanNeighIdx; ++nIdx)
	{
		neighbor = boundaryStrategy->getNeighborDirect(
			const_cast<Point3D&>(changePixel), nIdx);
		if (!neighbor.distance) {
			//if distance is 0 then the neighbor returned is invalid
			continue;
		}

		if (changeCell == cellFieldG->get(neighbor.pt)) ++neig_px_same_cell;
	}


	for (unsigned int nIdx = 0; nIdx <= fMooreNeighIdx; ++nIdx) //this loop might be useless
	{
		neighbor = boundaryStrategy->getNeighborDirect(
			const_cast<Point3D&>(changePixel), nIdx);
		if (!neighbor.distance) {
			//if distance is 0 then the neighbor returned is invalid
			continue;
		}
		search_domain.push_back(neighbor.pt);
	}

	if (neig_px_same_cell > 1)
	{
		bool N, E, S, W, NE, NW, SW, SE; // a flag for each cardinal position, true if that pixel bellongs to changeCell too.
		Point3D N_pt, E_pt, S_pt, W_pt, NE_pt, NW_pt, SW_pt, SE_pt;

		N_pt.y += 1; E_pt.x += 1; S_pt.y -= 1; W_pt.x -= 1; //still need to see the proper method for going trough border
		NE_pt.y += 1; NE_pt.x += 1;
		NW_pt.y += 1; NW_pt.x -= 1;
		SW_pt.y -= 1; SW_pt.x -= 1;
		SE_pt.y -= 1; SE_pt.x += 1;

		N = (cellFieldG->get(N_pt) == changeCell);
		E = (cellFieldG->get(E_pt) == changeCell);
		S = (cellFieldG->get(S_pt) == changeCell);
		W = (cellFieldG->get(W_pt) == changeCell);

		NE = (cellFieldG->get(NE_pt) == changeCell);
		NW = (cellFieldG->get(NW_pt) == changeCell);
		SW = (cellFieldG->get(SW_pt) == changeCell);
		SE = (cellFieldG->get(SE_pt) == changeCell);
		/*
		*neig_px_same_cell = 1 -> alway true
		* If neig_px_same_cell=2: If they are opposed, not locally connected.
		* 					 If they are corner-adjacent (e.g., West and North),
		*                  the north-West site must also have same value to have local connectivity.
		* If neig_px_same_cell=3: We look for the one which is not identical; if the
		*                 two opposed sites in the corners have the same value
		*                 as the central one, local connectivity is satisfied. Otherwise, it is not.
		* need to think about 4.
		*/
		if ((neig_px_same_cell == 2 && !((N && E && NE) || (N && W && NW) || (S && E && SE) || (S && W && SW))) ||
			(neig_px_same_cell == 3 && !((S || (NE&&NW)) && (E || (NW&&SW)) && (N || (SE&&SW)) && (W || (SE&&NE)))))
		{
			localConected = false;
		}

	}

	return localConected;



//
//
//
//	for (unsigned int nIdx = 0; nIdx <= fMooreNeighIdx; ++nIdx)//this loop might be useless
//	{
//		neighbor = boundaryStrategy->getNeighborDirect(
//			const_cast<Point3D&>(changePixel), nIdx);
//		if (!neighbor.distance) {
//			//if distance is 0 then the neighbor returned is invalid
//			continue;
//		}
//		nCell = cellFieldG->get(neighbor.pt);
//
//		if (!nCell)//if it's  medium we don't care
//		{
//			continue;
//		}
//
//
//
//
//		if (nCell == changeCell || nCell == flipCell) //since we've already know the source-target cells we don't need to check for others
//		{
//
//			//north => changePixel.y+1 == pt.y; x=x
//			//south => changePixel.y-1 == pt.y; x=x
//
//			//east => changePixel.x+1 == pt.x; y=y
//			//west => changePixel.x-1 == pt.x; y=y
//			if (changePixel.x == neighbor.pt.x)//north-south case
//			{
//
//				//might have issues with borders
//
//				Point3D right_pt = neighbor.pt;
//				right_pt.x += 1;
//
//				bool valid_right = boundaryStrategy->isValid(right_pt);//still need the check for across boundary
//
//				bool valid_right = (boundaryStrategy->isValid(right_pt) ? true : strategy_x->applyCondition(x, dim.x));
//
//				if (std::any_of(search_domain.begin(), search_domain.end(), right_pt) &&
//					valid_right)
//				{
//					if (nCell == cellFieldG->get(right_pt))
//					{
//						bool right_conceted = true;
//					}
//					else
//					{
//						bool right_conceted = false;
//					}
//
//				}
//
//
//
//				Point3D left_pt = neighbor.pt;
//				left_pt.x -= 1;
//
//				bool valid_left = boundaryStrategy->isValid(left_pt);//still need the check for across boundary
//				if (std::any_of(search_domain.begin(), search_domain.end(), left_pt) &&
//					valid_left)
//				{
//					if (nCell == cellFieldG->get(left_pt))
//					{
//						bool left_conceted = true;
//					}
//					else
//					{
//						bool left_conceted = false;
//					}
//
//				}
//
//				if (!(right_conceted || left_conceted))
//				{
//
//				}
//
//
//				/*if (!(nCell == cellFieldG->get(right_pt) || nCell == cellFieldG->get(left_pt)))
//				{
//					localConected = false;
//					return localConected;
//				}
//*/
//
//			}
//			else if (changePixel.y == neighbor.pt.y)//east-west case
//			{
//				//might have issues with borders
//				Point3D up_pt = neighbor.pt;
//				up_pt.y += 1;
//				Point3D down_pt = neighbor.pt;
//				down_pt.y -= 1;
//
//				if (!(nCell == cellFieldG->get(up_pt) || nCell == cellFieldG->get(down_pt)))
//				{
//					localConected = false;
//					return localConected;
//				}
//
//
//			}
//
//			//corner cases
//			else if (changePixel.y + 1 == neighbor.pt.y)
//			{
//				if (changePixel.x + 1 == neighbor.pt.x)//upper right corner
//				{
//					Point3D left_pt = neighbor.pt;
//					left_pt.x -= 1;
//					Point3D down_pt = neighbor.pt;
//					down_pt.y -= 1;
//					if (!(nCell == cellFieldG->get(left_pt) || nCell == cellFieldG->get(down_pt)))
//					{
//						return localConected = false;
//					}
//				}
//				else if (changePixel.x - 1 == neighbor.pt.x)//upper left corner
//				{
//					Point3D right_pt = neighbor.pt;
//					right_pt.x += 1;
//					Point3D down_pt = neighbor.pt;
//					down_pt.y -= 1;
//					if (!(nCell == cellFieldG->get(right_pt) || nCell == cellFieldG->get(down_pt)))
//					{
//						return localConected = false;
//					}
//				}
//
//			}
//
//			else if (changePixel.y - 1 == neighbor.pt.y)
//			{
//				if (changePixel.x + 1 == neighbor.pt.x)//lower right corner
//				{
//					Point3D left_pt = neighbor.pt;
//					left_pt.x -= 1;
//					Point3D up_pt = neighbor.pt;
//					up_pt.y += 1;
//					if (!(nCell == cellFieldG->get(left_pt) || nCell == cellFieldG->get(up_pt)))
//					{
//						return localConected = false;
//					}
//				}
//				else if (changePixel.x - 1 == neighbor.pt.x)//lower left corner
//				{
//					Point3D right_pt = neighbor.pt;
//					right_pt.x += 1;
//					Point3D up_pt = neighbor.pt;
//					up_pt.y += 1;
//					if (!(nCell == cellFieldG->get(right_pt) || nCell == cellFieldG->get(up_pt)))
//					{
//						return localConected = false;
//					}
//				}
//
//			}
//
//
//
//		}
//	}
//
//	return localConected;
}



bool Potts3D::localConect3D(Point3D changePixel, Point3D flipNeighbor)
{

	/*
	* Function to check the local conectivity of a cell in the flip pixel.
	* Based on https://doi.org/10.1016/j.cpc.2016.07.030
	* Durand, Marc, and Etienne Guesnet. "An efficient Cellular Potts Model
	* algorithm that forbids cell fragmentation." Computer Physics
	* Communications 208 (2016): 54-63.
	*
	* Change pixel is the pixel being changed. The value being copied to change
	* pixel in the one of flip neighbor.
	*
	* The algorithm only cares about the imediate vicinity (Moore neighb, order II,
	* all 8 - 2d -  touching pixels) of changePixel.
	*
	* The idea is to get all of the pixels in the neighborhood first. I do this
	* because I have to check some neighbors of the neighbor pixel. I won't cicle
	* through all of the neighbors' neighbors, just check a few. But when near the
	* border I could be picking an invalid pixel, get neighbor direct takes care
	* of checking valied pixels.
	*
	* However, I still will need to figure out the periodic boundary case. How to go to the
	* other side of the lattice eficiently? Better yet, with some method that has been written?
	*/

	//change goes from flip to change


	bool localConected = true;


	CellG *changeCell = cellFieldG->get(changePixel);

	CellG *flipCell = cellFieldG->get(flipNeighbor);

	CellG *nCell = 0;

	Neighbor neighbor;



	// check conectivity

	for (unsigned int nIdx = 0; nIdx <= 2; ++nIdx)
	{
		neighbor = boundaryStrategy->getNeighborDirect(
			const_cast<Point3D&>(changePixel), nIdx);
		if (!neighbor.distance) {
			//if distance is 0 then the neighbor returned is invalid
			continue;
		}
		nCell = cellFieldG->get(neighbor.pt);

		if (!nCell)//if it's  medium we don't care
		{
			continue;
		}


		if (nCell == changeCell || nCell == flipCell)
		{
			//changePixel.x == neighbor.pt.x

			if (changePixel.z == neighbor.pt.z)
			{
				if (changePixel.x == neighbor.pt.x)//north-south case
				{


					//might have issues with borders
					Point3D right_pt = neighbor.pt;
					right_pt.x += 1;
					Point3D left_pt = neighbor.pt;
					left_pt.x -= 1;

					Point3D up_pt = neighbor.pt;
					up_pt.z += 1;
					Point3D down_pt = neighbor.pt;
					down_pt.z -= 1;



					if (!(nCell == cellFieldG->get(right_pt) ||
						nCell == cellFieldG->get(left_pt) ||
						nCell == cellFieldG->get(up_pt) ||
						nCell == cellFieldG->get(down_pt)
						))
					{
						return localConected = false;
					}
				}

				else if (changePixel.y == neighbor.pt.y)//east-west case
				{
					//might have issues with borders
					Point3D right_pt = neighbor.pt;
					right_pt.y += 1;
					Point3D left_pt = neighbor.pt;
					left_pt.y -= 1;

					Point3D up_pt = neighbor.pt;
					up_pt.z += 1;
					Point3D down_pt = neighbor.pt;
					down_pt.z -= 1;

					if (!(nCell == cellFieldG->get(right_pt) ||
						nCell == cellFieldG->get(left_pt) ||
						nCell == cellFieldG->get(up_pt) ||
						nCell == cellFieldG->get(down_pt)
						))
					{
						return localConected = false;
					}
				}

				else if (changePixel.y + 1 == neighbor.pt.y)
				{
					if (changePixel.x + 1 == neighbor.pt.x)//upper right corner
					{
						Point3D left_pt = neighbor.pt;
						left_pt.x -= 1;
						Point3D backward_pt = neighbor.pt;
						backward_pt.y -= 1;

						Point3D up_pt = neighbor.pt;
						up_pt.z += 1;
						Point3D down_pt = neighbor.pt;
						down_pt.z -= 1;

						if (!(nCell == cellFieldG->get(backward_pt) ||
							nCell == cellFieldG->get(left_pt) ||
							nCell == cellFieldG->get(up_pt) ||
							nCell == cellFieldG->get(down_pt)
							))
						{
							return localConected = false;
						}
					}
					else if (changePixel.x - 1 == neighbor.pt.x)//upper right corner
					{
						Point3D right_pt = neighbor.pt;
						right_pt.x += 1;
						Point3D backward_pt = neighbor.pt;
						backward_pt.y -= 1;

						Point3D up_pt = neighbor.pt;
						up_pt.z += 1;
						Point3D down_pt = neighbor.pt;
						down_pt.z -= 1;

						if (!(nCell == cellFieldG->get(backward_pt) ||
							nCell == cellFieldG->get(right_pt) ||
							nCell == cellFieldG->get(up_pt) ||
							nCell == cellFieldG->get(down_pt)
							))
						{
							return localConected = false;
						}
					}
				}
				else if (changePixel.y - 1 == neighbor.pt.y)
				{
					if (changePixel.x + 1 == neighbor.pt.x)//upper right corner
					{
						Point3D left_pt = neighbor.pt;
						left_pt.x -= 1;
						Point3D forward_pt = neighbor.pt;
						forward_pt.y += 1;

						Point3D up_pt = neighbor.pt;
						up_pt.z += 1;
						Point3D down_pt = neighbor.pt;
						down_pt.z -= 1;

						if (!(nCell == cellFieldG->get(forward_pt) ||
							nCell == cellFieldG->get(left_pt) ||
							nCell == cellFieldG->get(up_pt) ||
							nCell == cellFieldG->get(down_pt)
							))
						{
							return localConected = false;
						}
					}
					else if (changePixel.x - 1 == neighbor.pt.x)//upper right corner
					{
						Point3D right_pt = neighbor.pt;
						right_pt.x += 1;
						Point3D forward_pt = neighbor.pt;
						forward_pt.y += 1;

						Point3D up_pt = neighbor.pt;
						up_pt.z += 1;
						Point3D down_pt = neighbor.pt;
						down_pt.z -= 1;

						if (!(nCell == cellFieldG->get(forward_pt) ||
							nCell == cellFieldG->get(right_pt) ||
							nCell == cellFieldG->get(up_pt) ||
							nCell == cellFieldG->get(down_pt)
							))
						{
							return localConected = false;
						}
					}
				}
			}
			else if (changePixel.z + 1 == neighbor.pt.z)
			{
				if (changePixel.x == neighbor.pt.x)//north-south case
				{


					//might have issues with borders
					Point3D right_pt = neighbor.pt;
					right_pt.x += 1;
					Point3D left_pt = neighbor.pt;
					left_pt.x -= 1;


					Point3D down_pt = neighbor.pt;
					down_pt.z -= 1;



					if (!(nCell == cellFieldG->get(right_pt) ||
						nCell == cellFieldG->get(left_pt) ||
						nCell == cellFieldG->get(down_pt)
						))
					{
						return localConected = false;
					}
				}

				else if (changePixel.y == neighbor.pt.y)//east-west case
				{
					//might have issues with borders
					Point3D right_pt = neighbor.pt;
					right_pt.y += 1;
					Point3D left_pt = neighbor.pt;
					left_pt.y -= 1;


					Point3D down_pt = neighbor.pt;
					down_pt.z -= 1;

					if (!(nCell == cellFieldG->get(right_pt) ||
						nCell == cellFieldG->get(left_pt) ||
						nCell == cellFieldG->get(down_pt)
						))
					{
						return localConected = false;
					}
				}

				else if (changePixel.y + 1 == neighbor.pt.y)
				{
					if (changePixel.x + 1 == neighbor.pt.x)//upper right corner
					{
						Point3D left_pt = neighbor.pt;
						left_pt.x -= 1;
						Point3D backward_pt = neighbor.pt;
						backward_pt.y -= 1;


						Point3D down_pt = neighbor.pt;
						down_pt.z -= 1;

						if (!(nCell == cellFieldG->get(backward_pt) ||
							nCell == cellFieldG->get(left_pt) ||
							nCell == cellFieldG->get(down_pt)
							))
						{
							return localConected = false;
						}
					}
					else if (changePixel.x - 1 == neighbor.pt.x)//upper right corner
					{
						Point3D right_pt = neighbor.pt;
						right_pt.x += 1;
						Point3D backward_pt = neighbor.pt;
						backward_pt.y -= 1;

						Point3D down_pt = neighbor.pt;
						down_pt.z -= 1;

						if (!(nCell == cellFieldG->get(backward_pt) ||
							nCell == cellFieldG->get(right_pt) ||
							nCell == cellFieldG->get(down_pt)
							))
						{
							return localConected = false;
						}
					}
				}
				else if (changePixel.y - 1 == neighbor.pt.y)
				{
					if (changePixel.x + 1 == neighbor.pt.x)//upper right corner
					{
						Point3D left_pt = neighbor.pt;
						left_pt.x -= 1;
						Point3D forward_pt = neighbor.pt;
						forward_pt.y += 1;

						Point3D down_pt = neighbor.pt;
						down_pt.z -= 1;

						if (!(nCell == cellFieldG->get(forward_pt) ||
							nCell == cellFieldG->get(left_pt) ||
							nCell == cellFieldG->get(down_pt)
							))
						{
							return localConected = false;
						}
					}
					else if (changePixel.x - 1 == neighbor.pt.x)//upper right corner
					{
						Point3D right_pt = neighbor.pt;
						right_pt.x += 1;
						Point3D forward_pt = neighbor.pt;
						forward_pt.y += 1;

						Point3D down_pt = neighbor.pt;
						down_pt.z -= 1;

						if (!(nCell == cellFieldG->get(forward_pt) ||
							nCell == cellFieldG->get(right_pt) ||
							nCell == cellFieldG->get(down_pt)
							))
						{
							return localConected = false;
						}
					}
				}
			}
			else if (changePixel.z - 1 == neighbor.pt.z)
			{
				if (changePixel.x == neighbor.pt.x)//north-south case
				{


					//might have issues with borders
					Point3D right_pt = neighbor.pt;
					right_pt.x += 1;
					Point3D left_pt = neighbor.pt;
					left_pt.x -= 1;

					Point3D up_pt = neighbor.pt;
					up_pt.z += 1;



					if (!(nCell == cellFieldG->get(right_pt) ||
						nCell == cellFieldG->get(left_pt) ||
						nCell == cellFieldG->get(up_pt)
						))
					{
						return localConected = false;
					}
				}

				else if (changePixel.y == neighbor.pt.y)//east-west case
				{
					//might have issues with borders
					Point3D right_pt = neighbor.pt;
					right_pt.y += 1;
					Point3D left_pt = neighbor.pt;
					left_pt.y -= 1;

					Point3D up_pt = neighbor.pt;
					up_pt.z += 1;

					if (!(nCell == cellFieldG->get(right_pt) ||
						nCell == cellFieldG->get(left_pt) ||
						nCell == cellFieldG->get(up_pt)
						))
					{
						return localConected = false;
					}
				}

				else if (changePixel.y + 1 == neighbor.pt.y)
				{
					if (changePixel.x + 1 == neighbor.pt.x)//upper right corner
					{
						Point3D left_pt = neighbor.pt;
						left_pt.x -= 1;
						Point3D backward_pt = neighbor.pt;
						backward_pt.y -= 1;

						Point3D up_pt = neighbor.pt;
						up_pt.z += 1;

						if (!(nCell == cellFieldG->get(backward_pt) ||
							nCell == cellFieldG->get(left_pt) ||
							nCell == cellFieldG->get(up_pt)
							))
						{
							return localConected = false;
						}
					}
					else if (changePixel.x - 1 == neighbor.pt.x)//upper right corner
					{
						Point3D right_pt = neighbor.pt;
						right_pt.x += 1;
						Point3D backward_pt = neighbor.pt;
						backward_pt.y -= 1;

						Point3D up_pt = neighbor.pt;
						up_pt.z += 1;

						if (!(nCell == cellFieldG->get(backward_pt) ||
							nCell == cellFieldG->get(right_pt) ||
							nCell == cellFieldG->get(up_pt)
							))
						{
							return localConected = false;
						}
					}
				}
				else if (changePixel.y - 1 == neighbor.pt.y)
				{
					if (changePixel.x + 1 == neighbor.pt.x)//upper right corner
					{
						Point3D left_pt = neighbor.pt;
						left_pt.x -= 1;
						Point3D forward_pt = neighbor.pt;
						forward_pt.y += 1;

						Point3D up_pt = neighbor.pt;
						up_pt.z += 1;

						if (!(nCell == cellFieldG->get(forward_pt) ||
							nCell == cellFieldG->get(left_pt) ||
							nCell == cellFieldG->get(up_pt)
							))
						{
							return localConected = false;
						}
					}
					else if (changePixel.x - 1 == neighbor.pt.x)//upper right corner
					{
						Point3D right_pt = neighbor.pt;
						right_pt.x += 1;
						Point3D forward_pt = neighbor.pt;
						forward_pt.y += 1;

						Point3D up_pt = neighbor.pt;
						up_pt.z += 1;

						if (!(nCell == cellFieldG->get(forward_pt) ||
							nCell == cellFieldG->get(right_pt) ||
							nCell == cellFieldG->get(up_pt)
							))
						{
							return localConected = false;
						}
					}
				}
			}

		}
	}

	return localConected;
}


bool Potts3D::localConectHex(Point3D changePixel, Point3D flipNeighbor)
{

	/*
	* Function to check the local conectivity of a cell in the flip pixel.
	* Based on https://doi.org/10.1016/j.cpc.2016.07.030
	* Durand, Marc, and Etienne Guesnet. "An efficient Cellular Potts Model
	* algorithm that forbids cell fragmentation." Computer Physics
	* Communications 208 (2016): 54-63.
	*
	* Change pixel is the pixel being changed. The value being copied to change
	* pixel in the one of flip neighbor.
	*
	* The algorithm only cares about the imediate vicinity (Moore neighb, order II,
	* all 8 - 2d -  touching pixels) of changePixel.
	*
	* The idea is to get all of the pixels in the neighborhood first. I do this
	* because I have to check some neighbors of the neighbor pixel. I won't cicle
	* through all of the neighbors' neighbors, just check a few. But when near the
	* border I could be picking an invalid pixel, get neighbor direct takes care
	* of checking valied pixels.
	*
	* However, I still will need to figure out the periodic boundary case. How to go to the
	* other side of the lattice eficiently? Better yet, with some method that has been written?
	*/

	return true; //PLACEHODLER
}

void Potts3D::setAcceptanceFunctionByName(std::string _acceptanceFunctionName) {
	if (_acceptanceFunctionName == "FirstOrderExpansion") {
		acceptanceFunction = &firstOrderExpansionAcceptanceFunction;
		//          cerr<<"setting FirstOrderExpansion"<<endl;
	}
	else {
		acceptanceFunction = &defaultAcceptanceFunction;
	}

}

void Potts3D::registerAcceptanceFunction(AcceptanceFunction *function) {
	ASSERT_OR_THROW("registerAcceptanceFunction() function cannot be NULL!",
		function);

	acceptanceFunction = function;
}


void  Potts3D::setFluctuationAmplitudeFunctionByName(std::string _fluctuationAmplitudeFunctionName) {

	if (_fluctuationAmplitudeFunctionName == "Min") {

		delete fluctAmplFcn;
		fluctAmplFcn = new MinFluctuationAmplitudeFunction(this);

	}
	else if (_fluctuationAmplitudeFunctionName == "Max") {

		delete fluctAmplFcn;
		fluctAmplFcn = new MaxFluctuationAmplitudeFunction(this);

	}
	else if (_fluctuationAmplitudeFunctionName == "ArithmeticAverage") {

		delete fluctAmplFcn;
		fluctAmplFcn = new ArithmeticAverageFluctuationAmplitudeFunction(this);

	}
}

///BasicClassChange watcher reistration
void Potts3D::registerCellGChangeWatcher(CellGChangeWatcher *_watcher) {
	ASSERT_OR_THROW("registerBCGChangeWatcher() _watcher cannot be NULL!", _watcher);

	cellFieldG->addChangeWatcher(_watcher);
	//    sim->registerSteerableObject(_watcher);
}


void Potts3D::registerClassAccessor(BasicClassAccessorBase *_accessor) {
	ASSERT_OR_THROW("registerClassAccessor() _accessor cannot be NULL!", _accessor);

	cellFactoryGroup.registerClass(_accessor);
}


void Potts3D::registerStepper(Stepper *stepper) {
	ASSERT_OR_THROW("registerStepper() stepper cannot be NULL!", stepper);

	steppers.push_back(stepper);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Potts3D::registerFixedStepper(FixedStepper *fixedStepper, bool _front) {
	ASSERT_OR_THROW("registerStepper() fixed stepper cannot be NULL!", fixedStepper);
	if (_front) {
		//fixedSteppers is small so using deque as temporary storage to insert at the begining of vector is OK
		deque<FixedStepper *> tmpDeque(fixedSteppers.begin(), fixedSteppers.end());
		tmpDeque.push_front(fixedStepper);
		fixedSteppers = vector<FixedStepper *>(tmpDeque.begin(), tmpDeque.end());
	}
	else {

		fixedSteppers.push_back(fixedStepper);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Potts3D::unregisterFixedStepper(FixedStepper *_fixedStepper) {
	ASSERT_OR_THROW("unregisterStepper() fixed stepper cannot be NULL!", _fixedStepper);
	std::vector<FixedStepper *>::iterator pos;
	pos = find(fixedSteppers.begin(), fixedSteppers.end(), _fixedStepper);
	if (pos != fixedSteppers.end()) {
		fixedSteppers.erase(pos);
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CellG * Potts3D::createCellG(const Point3D pt, long _clusterId) {
	ASSERT_OR_THROW("createCell() cellFieldG Point out of range!", cellFieldG->isValid(pt));

	CellG * cell = createCell(_clusterId);

	cellFieldG->set(pt, cell);

	return cell;

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CellG *Potts3D::createCell(long _clusterId) {
	CellG * cell = new CellG();
	cell->extraAttribPtr = cellFactoryGroup.create();
	//always keep incrementing recently created  cell id
	++recentlyCreatedCellId;
	cell->id = recentlyCreatedCellId;


	//this means that cells with clusterId<=0 should be placed at the end of PIF file if automatic numbering of clusters is to work for a mix of clustered and non clustered cells	

	if (_clusterId <= 0) { //default behavior if user does not specify cluster id or cluster id is 0
		++recentlyCreatedClusterId;
		cell->clusterId = recentlyCreatedClusterId;


	}
	else if (_clusterId > recentlyCreatedClusterId) { //clusterId specified by user is greater than recentlyCreatedClusterId

		cell->clusterId = _clusterId;
		// if we get cluster id greater than recentlyCreatedClusterId we set recentlyCreatedClusterId to be _clusterId+1
		// this way if users add "non-cluster" cells after definition of clustered cell 
		// the cell->clusterId is guaranteed to be greater than any of the clusterIds specified for clustered cells
		recentlyCreatedClusterId = _clusterId;


	}
	else { // cluster id is greater than zero but smaller than recentlyCreatedClusterId
		cell->clusterId = _clusterId;
	}

	cellInventory.addToInventory(cell);

	if (attrAdder) {
		attrAdder->addAttribute(cell);
	}
	return cell;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// this function should be only used from PIF Initializers or when you really understand the way CC3D assigns cell ids
CellG * Potts3D::createCellGSpecifiedIds(const Point3D pt, long _cellId, long _clusterId) {
	ASSERT_OR_THROW("createCell() cellFieldG Point out of range!", cellFieldG->isValid(pt));
	CellG * cell = createCellSpecifiedIds(_cellId, _clusterId);
	cellFieldG->set(pt, cell);
	return cell;

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// this function should be only used from PIF Initializers or when you really understand the way CC3D assigns cell ids
CellG *Potts3D::createCellSpecifiedIds(long _cellId, long _clusterId) {
	CellG * cell = new CellG();
	cell->extraAttribPtr = cellFactoryGroup.create();

	if (_cellId > recentlyCreatedCellId) {
		recentlyCreatedCellId = _cellId;
		cell->id = recentlyCreatedCellId;
	}
	else {
		// override _cellId and use recentlyCreatedCellId as _cellId. 
		// otherwise we may create a bug where cells initialized from e.g. PIF initializers will have same id as cells created earlier by e.g. uniform initializer
		// they will have different cluster id but still it creates a problem/. Having a model where cell id and cluster id are always incremented is a safer (but not ideal) bet 
		++recentlyCreatedCellId;
		cell->id = recentlyCreatedCellId;
	}

	//this means that cells with clusterId<=0 should be placed at the end of PIF file if automatic numbering of clusters is to work for a mix of clustered and non clustered cells	

	if (_clusterId <= 0) { //default behavior if user does not specify cluster id or cluster id is 0

		++recentlyCreatedClusterId;
		cell->clusterId = recentlyCreatedClusterId;

	}
	else if (_clusterId > recentlyCreatedClusterId) { //clusterId specified by user is greater than recentlyCreatedClusterId

		cell->clusterId = _clusterId;
		// if we get cluster id greater than recentlyCreatedClusterId we set recentlyCreatedClusterId to be _clusterId+1
		// this way if users add "non-cluster" cells after definition of clustered cells	the cell->clusterId is guaranteed to be greater than any of the clusterIds specified for clustered cells
		recentlyCreatedClusterId = _clusterId;

	}
	else { // cluster id is greater than zero but smaller than recentlyCreatedClusterId
		cell->clusterId = _clusterId;
	}

	cellInventory.addToInventory(cell);
	if (attrAdder) {
		attrAdder->addAttribute(cell);
	}
	return cell;

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Potts3D::destroyCellG(CellG *cell, bool _removeFromInventory) {
	if (cell->extraAttribPtr) {
		cellFactoryGroup.destroy(cell->extraAttribPtr);
		cell->extraAttribPtr = 0;
	}
	if (cell->pyAttrib && attrAdder) {
		attrAdder->destroyAttribute(cell);
	}
	//had to introduce these two cases because in the Cell inventory destructor we deallocate memory of pointers stored int the set
	//Since this is done during interation over set changing pointer (cell=0) or removing anything from set might corrupt container or invalidate iterators
	if (_removeFromInventory) {
		cellInventory.removeFromInventory(cell);
		delete cell;
		cell = 0;
	}
	else {
		delete cell;
	}

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double Potts3D::totalEnergy() {
	double energy = 0;
	Dim3D dim = cellFieldG->getDim();

	Point3D pt;
	for (pt.z = 0; pt.z < dim.z; pt.z++)
		for (pt.y = 0; pt.y < dim.y; pt.y++)
			for (pt.x = 0; pt.x < dim.x; pt.x++)
				for (unsigned int i = 0; i < energyFunctions.size(); i++)
					energy += energyFunctions[i]->localEnergy(pt);

	return energy;

}


double Potts3D::changeEnergy(Point3D pt, const CellG *newCell,
	const CellG *oldCell) {
	double change = 0;
	for (unsigned int i = 0; i < energyFunctions.size(); i++)
		change += energyFunctions[i]->changeEnergy(pt, newCell, oldCell);

	return change;
}

void Potts3D::runSteppers() {
	for (unsigned int j = 0; j < steppers.size(); j++)
		steppers[j]->step();
}

unsigned int Potts3D::metropolis(const unsigned int steps, const double temp) {
	temperature = temp;
	return (this->*metropolisFcnPtr)(steps, temp);
}

unsigned int Potts3D::metropolisList(const unsigned int steps, const double temp) {
	cerr << "metropolisList" << endl;
	ASSERT_OR_THROW("Potts3D: cell field G not initialized", cellFieldG);

	// ParallelUtilsOpenMP * pUtils=sim->getParallelUtils();


	ASSERT_OR_THROW("MetropolisList algorithm works only in single processor mode. Please change number of processors to 1", pUtils->getNumberOfWorkNodesPotts() == 1);

	if (customAcceptanceExpressionDefined) {
		customAcceptanceFunction.initialize(this->sim); //actual initialization will happen only once at MCS=0 all other calls will return without doing anything
	}

	//here we will allocate Random number generators for each thread. Note that since user may change number of work nodes we have to monitor if the max number of work threads is greater than size of random number generator vector 
	if (!randNSVec.size() || pUtils->getMaxNumberOfWorkNodesPotts() > randNSVec.size()) { //each thread will have different random number ghenerator
		randNSVec.assign(pUtils->getMaxNumberOfWorkNodesPotts(), BasicRandomNumberGeneratorNonStatic());

		for (unsigned int i = 0; i < randNSVec.size(); ++i) {
			if (!sim->ppdCC3DPtr->seed) {
				srand(time(0));
				unsigned int randomSeed = (unsigned int)rand()*((std::numeric_limits<unsigned int>::max)() - 1);
				randNSVec[i].setSeed(randomSeed);
			}
			else {
				randNSVec[i].setSeed(sim->ppdCC3DPtr->seed);
			}
		}
	}
	// Note that since user may change number of work nodes we have to monitor if the max number of work threads is greater than size of flipNeighborVec
	if (!flipNeighborVec.size() || pUtils->getMaxNumberOfWorkNodesPotts() > flipNeighborVec.size()) {
		flipNeighborVec.assign(pUtils->getMaxNumberOfWorkNodesPotts(), Point3D());
	}
	flips = 0;
	attemptedEC = 0;
	BasicRandomNumberGenerator *rand = BasicRandomNumberGenerator::getInstance();
	///Dim3D dim = cellField->getDim();
	Dim3D dim = cellFieldG->getDim();
	ASSERT_OR_THROW("Potts3D: You must supply an acceptance function!",
		acceptanceFunction);

	//   numberOfAttempts=steps;
	numberOfAttempts = (int)(maxCoordinates.x - minCoordinates.x)*(maxCoordinates.y - minCoordinates.y)*(maxCoordinates.z - minCoordinates.z)*sim->getFlip2DimRatio()*sim->getFlip2DimRatio();

	BoundaryStrategy * boundaryStrategy = BoundaryStrategy::getInstance();
	pUtils->prepareParallelRegionPotts();
	pUtils->allowNestedParallelRegions(true); //necessary in case we use e.g. PDE solver caller which in turn calls parallel PDE solver

#pragma omp parallel
	{

		for (unsigned int i = 0; i < numberOfAttempts; i++) {

			currentAttempt = i;
			//run fixed steppers - they are executed regardless whether spin flip take place or not . Note, regular steopers are executed only after spin flip attepmts takes place 
			for (unsigned int j = 0; j < fixedSteppers.size(); j++)
				fixedSteppers[j]->step();

			Point3D pt;
			// Pick a random point
			pt.x = rand->getInteger(minCoordinates.x, maxCoordinates.x - 1);
			pt.y = rand->getInteger(minCoordinates.y, maxCoordinates.y - 1);
			pt.z = rand->getInteger(minCoordinates.z, maxCoordinates.z - 1);

			///Cell *cell = cellField->get(pt);
			CellG *cell = cellFieldG->get(pt);

			if (sizeFrozenTypeVec && cell) {///must also make sure that cell ptr is different 0; Will never freeze medium
				if (checkIfFrozen(cell->type))
					continue;
			}

			unsigned int token = 0;
			int numNeighbors = 0;
			double distance;

			token = 0;
			distance = 0;
			while (true) {

				neighbors[numNeighbors] = cellFieldG->getNeighbor(pt, token, distance);
				if (distance > depth) break;

				if (cellFieldG->get(neighbors[numNeighbors]) != cell)
					numNeighbors++;

			}

			// If no boundry neighbors were found start over
			if (!numNeighbors) continue;

			// Pick a random neighbor
			Point3D changePixel = neighbors[rand->getInteger(0, numNeighbors - 1)];




			if (sizeFrozenTypeVec && cellFieldG->get(changePixel)) {///must also make sure that cell ptr is different 0; Will never freeze medium
				if (checkIfFrozen(cellFieldG->get(changePixel)->type))
					continue;
			}


			++attemptedEC;

			flipNeighbor = pt;/// change takes place at change pixel  and pt is a neighbor of changePixel

			//I could also do the conectivity check here, still has the possible conflict with multithread?





			// Calculate change in energy

			double change = energyCalculator->changeEnergy(changePixel, cell, cellFieldG->get(changePixel), i);
			///cerr<<"This is change: "<<change<<endl;	

			// Acceptance based on probability
			double motility = fluctAmplFcn->fluctuationAmplitude(cell, cellFieldG->get(changePixel));
			//double motility=0.0;
			//if(cellTypeMotilityVec.size()){
			//	unsigned int newCellTypeId=(cell ? (unsigned int)cell->type :0);
			//	unsigned int oldCellTypeId=(cellFieldG->get(changePixel)? (unsigned int)cellFieldG->get(changePixel)->type :0);
			//	if(newCellTypeId && oldCellTypeId)
			//		motility=(cellTypeMotilityVec[newCellTypeId]<cellTypeMotilityVec[oldCellTypeId] ? cellTypeMotilityVec[newCellTypeId]:cellTypeMotilityVec[oldCellTypeId]);
			//	else if(newCellTypeId){
			//		motility=cellTypeMotilityVec[newCellTypeId];
			//	}else if (oldCellTypeId){
			//		motility=cellTypeMotilityVec[oldCellTypeId];
			//	}else{//should never get here
			//		motility=0;
			//	}
			//}else{
			//	motility=temp;
			//}
			double prob = acceptanceFunction->accept(motility, change);



			if (prob >= 1 || rand->getRatio() < prob) {
				// Accept the change
				energy += change;

				if (connectivityConstraint && connectivityConstraint->changeEnergy(changePixel, cell, cellFieldG->get(changePixel))) {
					//put call to durand fragmentation in this if
					// new if:

					//if (connectivityConstraint || durandContraint)
					//{ if(connectivityConstraint && connectivityConstraint->changeEnergy(changePixel, cell, changePixelCell))
					//  {...}
					//  if(durandContraint)
					//  {...}
					energyCalculator->setLastFlipAccepted(false);
				}
				else {
					cellFieldG->set(changePixel, cell);
					flips++;
					energyCalculator->setLastFlipAccepted(true);
				}
			}
			else {
				energyCalculator->setLastFlipAccepted(false);
			}


			// Run steppers
			for (unsigned int j = 0; j < steppers.size(); j++)
				steppers[j]->step();
		}
	}//#pragma omp parallel 
	unsigned int currentStep = sim->getStep();
	if (debugOutputFrequency && !(currentStep % debugOutputFrequency)) {
		cerr << "Number of Attempted Energy Calculations=" << attemptedEC << endl;
	}
	return flips;
}

Point3D Potts3D::getFlipNeighbor() {
	return flipNeighborVec[sim->getParallelUtils()->getCurrentWorkNodeNumber()];
}

unsigned int Potts3D::metropolisFast(const unsigned int steps, const double temp) {


	ASSERT_OR_THROW("Potts3D: cell field G not initialized", cellFieldG);
	// // // ParallelUtilsOpenMP * pUtils=sim->getParallelUtils();

	if (customAcceptanceExpressionDefined) {
		customAcceptanceFunction.initialize(this->sim); //actual initialization will happen only once at MCS=0 all other calls will return without doing anything
	}

	//here we will allocate Random number generators for each thread. Note that since user may change number of work nodes we have to monitor if the max number of work threads is greater than size of random number generator vector 
	if (!randNSVec.size() || pUtils->getMaxNumberOfWorkNodesPotts() > randNSVec.size()) { //each thread will have different random number ghenerator
		randNSVec.assign(pUtils->getMaxNumberOfWorkNodesPotts(), BasicRandomNumberGeneratorNonStatic());

		for (unsigned int i = 0; i < randNSVec.size(); ++i) {
			if (!sim->ppdCC3DPtr->seed) {
				srand(time(0));
				unsigned int randomSeed = (unsigned int)rand()*((std::numeric_limits<unsigned int>::max)() - 1);
				randNSVec[i].setSeed(randomSeed);
			}
			else {
				randNSVec[i].setSeed(sim->ppdCC3DPtr->seed);
			}
		}
	}


	// Note that since user may change number of work nodes we have to monitor if the max number of work threads is greater than size of flipNeighborVec
	if (!flipNeighborVec.size() || pUtils->getMaxNumberOfWorkNodesPotts() > flipNeighborVec.size()) {
		flipNeighborVec.assign(pUtils->getMaxNumberOfWorkNodesPotts(), Point3D());
	}

	// generating random order in which subgridSections will be handled
	vector<unsigned int> subgridSectionOrderVec(pUtils->getNumberOfSubgridSectionsPotts());
	for (int i = 0; i < subgridSectionOrderVec.size(); ++i) {
		subgridSectionOrderVec[i] = i;
	}
	random_shuffle(subgridSectionOrderVec.begin(), subgridSectionOrderVec.end());

	unsigned int maxNumberOfThreads = pUtils->getMaxNumberOfWorkNodesPotts();
	unsigned int numberOfThreads = pUtils->getNumberOfWorkNodesPotts();

	unsigned int numberOfSections = pUtils->getNumberOfSubgridSectionsPotts();

	currentAttempt = 0; //reset current attepmt counter

	//THESE VARIABLES ARE SHARED
	flips = 0;
	attemptedEC = 0;
	energy = 0.0;

	//THESE WILL BE USED IN SEPARATE THREADS - WE USE VECTORS TO AVOID USING SYNCHRONIZATION EXPLICITELY
	vector<double> energyVec(maxNumberOfThreads, 0.0);
	vector<int>	attemptedECVec(maxNumberOfThreads, 0);
	vector<int> flipsVec(maxNumberOfThreads, 0);

	Dim3D dim = cellFieldG->getDim();
	ASSERT_OR_THROW("Potts3D: You must supply an acceptance function!", acceptanceFunction);

	//FOR NOW WE WILL IGNORE BOX WATCHER FOR POTTS SECTION IT WILL STILL WORK WITH PDE SOLVERS
	Dim3D fieldDim = cellFieldG->getDim();
	numberOfAttempts = (int)fieldDim.x*fieldDim.y*fieldDim.z*sim->getFlip2DimRatio();
	unsigned int currentStep = sim->getStep();
	if (debugOutputFrequency && !(currentStep % debugOutputFrequency)) {
		cerr << "Metropolis Fast" << endl;
		cerr << "total number of pixel copy attempts=" << numberOfAttempts << endl;
	}

	pUtils->prepareParallelRegionPotts();
	pUtils->allowNestedParallelRegions(true); //necessary in case we use e.g. PDE solver caller which in turn calls parallel PDE solver
	//omp_set_nested(true);

#pragma omp parallel
	{

		int currentAttemptLocal;
		int numberOfAttemptsLocal;
		Point3D flipNeighborLocal;

		unsigned int currentWorkNodeNumber = pUtils->getCurrentWorkNodeNumber();

		BasicRandomNumberGeneratorNonStatic * rand = randNSVec[currentWorkNodeNumber].getInstance();
		BoundaryStrategy * boundaryStrategy = BoundaryStrategy::getInstance();

		//iterating over subgridSections
		for (int s = 0; s < subgridSectionOrderVec.size(); ++s) {

			pair<Dim3D, Dim3D> sectionDims = pUtils->getPottsSection(currentWorkNodeNumber, s);
			numberOfAttemptsLocal = (int)(sectionDims.second.x - sectionDims.first.x)*(sectionDims.second.y - sectionDims.first.y)*(sectionDims.second.z - sectionDims.first.z)*sim->getFlip2DimRatio();
			// #pragma omp critical


			for (unsigned int i = 0; i < numberOfAttemptsLocal; ++i) {

				//////currentAttempt=i;
				//currentAttemptLocal=i; //this will need to be fixed as this is used in PDE SOLVER CALLER
				//run fixed steppers - they are executed regardless whether spin flip take place or not . Note, regular stepers are executed only after spin flip attepmts takes place 

				if (fixedSteppers.size()) {
#pragma omp critical
					{
						//IMPORTANT: fixed steppers cause really bad performance with multiple processor runs. Currently only two of them are supported 
						// PDESolverCaller plugin and Secretion plugin. PDESolverCaller is deprecated and Secretion plugin section that mimics functionality of PDE sovler Secretion data section is depreecated as well
						// However Secretion plugin can be used to do "per cell" secretion (using python scripting). This will not slow down multicore simulation

						//cerr<<"pUtils->getCurrentWorkNodeNumber()="<<pUtils->getCurrentWorkNodeNumber()<<" currentAttempt="<<currentAttempt<<endl;
						for (unsigned int j = 0; j < fixedSteppers.size(); j++)
							fixedSteppers[j]->step();

						++currentAttempt; //to be consistent with serial code currentAttampt has to be increamented after fixedSteppers run 


					}
				}

				Point3D pt;

				// Pick a random point
				pt.x = rand->getInteger(sectionDims.first.x, sectionDims.second.x - 1);
				pt.y = rand->getInteger(sectionDims.first.y, sectionDims.second.y - 1);
				pt.z = rand->getInteger(sectionDims.first.z, sectionDims.second.z - 1);

				///Cell *cell = cellField->get(pt);
				CellG *cell = cellFieldG->getQuick(pt);


				if (sizeFrozenTypeVec && cell) {///must also make sure that cell ptr is different 0; Will never freeze medium
					if (checkIfFrozen(cell->type))
						continue;
				}

				unsigned int directIdx = rand->getInteger(0, maxNeighborIndex);


				Neighbor n = boundaryStrategy->getNeighborDirect(pt, directIdx);
				//if(currentWorkNodeNumber==0){

				//      cerr<<" pt="<<pt<<" n.pt"<<n.pt<<" n.distance="<<n.distance<<endl;
				//      cerr<<"directIdx="<<directIdx<<" maxNeighborIndex="<<maxNeighborIndex<<endl;
				//}


				if (!n.distance) {
					//if distance is 0 then the neighbor returned is invalid
					continue;
				}
				Point3D changePixel = n.pt;

				//check if changePixel refers to different cell. 
				CellG* changePixelCell = cellFieldG->getQuick(changePixel);

				if (changePixelCell == cell) {
					continue;//skip the rest of the loop if change pixel points to the same cell as pt
				}
				else {
					;

				}


				if (sizeFrozenTypeVec && changePixelCell) {///must also make sure that cell ptr is different 0; Will never freeze medium
					if (checkIfFrozen(changePixelCell->type))
						continue;
				}
				++attemptedECVec[currentWorkNodeNumber];

				flipNeighborVec[currentWorkNodeNumber] = pt;

				/// change takes place at change pixel  and pt is a neighbor of changePixel
				// Calculate change in energy				

				double change = energyCalculator->changeEnergy(changePixel, cell, changePixelCell, i);

				// Acceptance based on probability
				double motility = fluctAmplFcn->fluctuationAmplitude(cell, changePixelCell);

				double prob = acceptanceFunction->accept(motility, change);
				//#pragma omp critical
				//				{
				//				cerr<<" thread="<<currentWorkNodeNumber<<" section="<<s<<" prob="<<prob<<" energy="<<energy<<endl;
				//				}


				if (numberOfThreads == 1) {
					energyCalculator->set_aceptance_probability(prob);
				}

				//     cerr<<"change E="<<change<<" prob="<<prob<<endl;
				if (prob >= 1.0 || rand->getRatio() < prob) {
					// Accept the change

					energyVec[currentWorkNodeNumber] += change;

					if (connectivityConstraint && connectivityConstraint->changeEnergy(changePixel, cell, changePixelCell)) {
						//put call to durand fragmentation in this if
						// new if:

						//if (connectivityConstraint || durandContraint)
						//{ if(connectivityConstraint && connectivityConstraint->changeEnergy(changePixel, cell, changePixelCell))
						//  {...}
						//  if(durandContraint)
						//  {...}
						if (numberOfThreads == 1) {
							energyCalculator->setLastFlipAccepted(false);
						}
					}
					else {
						cellFieldG->set(changePixel, cell);
						flipsVec[currentWorkNodeNumber]++;
						if (numberOfThreads == 1) {
							energyCalculator->setLastFlipAccepted(true);
						}
					}
				}
				else {
					if (numberOfThreads == 1) {
						energyCalculator->setLastFlipAccepted(false);
					}
				}


				// Run steppers
//#pragma omp single 
				{
					for (unsigned int j = 0; j < steppers.size(); j++)
						steppers[j]->step();
				}

			}

#pragma omp barrier
		}//iteration over subrid sections

#pragma omp critical
		{
			//cerr<<"energyVec.size()="<<energyVec.size()<<endl;
			//cerr<<"attemptedECVec.size()="<<attemptedECVec.size()<<endl;
			//cerr<<"flipsVec.size()="<<flipsVec.size()<<endl;

			energy += energyVec[currentWorkNodeNumber];
			attemptedEC += attemptedECVec[currentWorkNodeNumber];
			flips += flipsVec[currentWorkNodeNumber];

			//cerr<<"thread "<<currentWorkNodeNumber<<" has finished calculations and did "<<attemptedECVec[currentWorkNodeNumber]<<" calculations"<<endl;

			//reseting values before processing new slice

			energyVec[currentWorkNodeNumber] = 0.0;
			attemptedECVec[currentWorkNodeNumber] = 0;
			flipsVec[currentWorkNodeNumber] = 0;

		}

	} //pragma omp parallel

	if (debugOutputFrequency && !(currentStep % debugOutputFrequency)) {
		cerr << "Number of Attempted Energy Calculations=" << attemptedEC << endl;
	}
	//cerr<<"CURRENT ATTEMPT="<<currentAttempt<<" numberOfAttempts="<<numberOfAttempts<<endl;
	//    exit(0);
	return flips;

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Point3D Potts3D::randomPickBoundaryPixel(BasicRandomNumberGeneratorNonStatic * rand) {

	size_t vec_size = boundaryPixelVector.size();
	Point3D pt;
	int counter = 0;
	while (true) {
		++counter;
		long boundaryPointIndex = rand->getInteger(0, boundaryPixelSet.size() - 1);
		if (boundaryPointIndex < vec_size) {
			pt = boundaryPixelVector[boundaryPointIndex];
		}
		else {
			std::unordered_set<Point3D, Point3DHasher, Point3DComparator>::iterator sitr = justInsertedBoundaryPixelSet.begin();
			advance(sitr, boundaryPointIndex - vec_size);
			pt = *sitr;

		}
		if (justDeletedBoundaryPixelSet.find(pt) != justDeletedBoundaryPixelSet.end()) {

		}
		else {
			break;
		}
	}
	if (counter > 5) {
		cerr << "had to try more than 5 times" << endl;
	}
	return pt;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
unsigned int Potts3D::metropolisBoundaryWalker(const unsigned int steps, const double temp) {



	ASSERT_OR_THROW("BoundaryWalker Algorithm works only in single processor mode. Please change number of processors to 1", pUtils->getNumberOfWorkNodesPotts() == 1);

	ASSERT_OR_THROW("Potts3D: cell field G not initialized", cellFieldG);

	if (customAcceptanceExpressionDefined) {
		customAcceptanceFunction.initialize(this->sim); //actual initialization will happen only once at MCS=0 all other calls will return without doing anything
	}

	//here we will allocate Random number generators for each thread. Note that since user may change number of work nodes we have to monitor if the max number of work threads is greater than size of random number generator vector 
	if (!randNSVec.size() || pUtils->getMaxNumberOfWorkNodesPotts() > randNSVec.size()) { //each thread will have different random number ghenerator
		randNSVec.assign(pUtils->getMaxNumberOfWorkNodesPotts(), BasicRandomNumberGeneratorNonStatic());

		for (unsigned int i = 0; i < randNSVec.size(); ++i) {
			if (!sim->ppdCC3DPtr->seed) {
				srand(time(0));
				unsigned int randomSeed = (unsigned int)rand()*((std::numeric_limits<unsigned int>::max)() - 1);
				randNSVec[i].setSeed(randomSeed);
			}
			else {
				randNSVec[i].setSeed(sim->ppdCC3DPtr->seed);
			}
		}
	}

	// Note that since user may change number of work nodes we have to monitor if the max number of work threads is greater than size of flipNeighborVec
	if (!flipNeighborVec.size() || pUtils->getMaxNumberOfWorkNodesPotts() > flipNeighborVec.size()) {
		flipNeighborVec.assign(pUtils->getMaxNumberOfWorkNodesPotts(), Point3D());
	}

	// generating random order in which subgridSections will be handled
	vector<unsigned int> subgridSectionOrderVec(pUtils->getNumberOfSubgridSectionsPotts());
	for (int i = 0; i < subgridSectionOrderVec.size(); ++i) {
		subgridSectionOrderVec[i] = i;
	}
	random_shuffle(subgridSectionOrderVec.begin(), subgridSectionOrderVec.end());


	unsigned int maxNumberOfThreads = pUtils->getMaxNumberOfWorkNodesPotts();
	unsigned int numberOfThreads = pUtils->getNumberOfWorkNodesPotts();

	unsigned int numberOfSections = pUtils->getNumberOfSubgridSectionsPotts();

	//reset current attepmt counter
	currentAttempt = 0;

	//THESE VARIABLES ARE SHARED
	flips = 0;
	attemptedEC = 0;
	energy = 0.0;

	//THESE WILL BE USED IN SEPARATE THREADS - WE USE VECTORS TO AVOID USING SYNCHRONIZATION EXPLICITELY
	vector<double> energyVec(maxNumberOfThreads, 0.0);
	vector<int>	attemptedECVec(maxNumberOfThreads, 0);
	vector<int> flipsVec(maxNumberOfThreads, 0);

	Dim3D dim = cellFieldG->getDim();
	ASSERT_OR_THROW("Potts3D: You must supply an acceptance function!",
		acceptanceFunction);

	//FOR NOW WE WILL IGNORE BOX WATCHER FOR POTTS SECTION IT WILL STILL WORK WITH PDE SOLVERS
	Dim3D fieldDim = cellFieldG->getDim();
	numberOfAttempts = (int)fieldDim.x*fieldDim.y*fieldDim.z*sim->getFlip2DimRatio();
	unsigned int currentStep = sim->getStep();


	pUtils->prepareParallelRegionPotts();

	//necessary in case we use e.g. PDE solver caller which in turn calls parallel PDE solver
	//omp_set_nested(true);
	pUtils->allowNestedParallelRegions(true);

	long boundaryPointIndex;
	long counter = 0;
	//set<Point3D>::iterator sitr;
	std::unordered_set<Point3D, Point3DHasher, Point3DComparator>::iterator sitr;
	vector<Point3D> boundaryPointVector;
	//boundaryPointVector.assign(boundaryPixelSet.begin(), boundaryPixelSet.end());
	numberOfAttempts = boundaryPixelSet.size();
	if (debugOutputFrequency && !(currentStep % debugOutputFrequency)) {
		cerr << "Boundary Walker" << endl;
		cerr << "number pixel copy attempts=" << numberOfAttempts << endl;
	}

#pragma omp parallel
	{

		int currentAttemptLocal;
		//int numberOfAttemptsLocal;
		Point3D flipNeighborLocal;

		unsigned int currentWorkNodeNumber = pUtils->getCurrentWorkNodeNumber();

		BasicRandomNumberGeneratorNonStatic * rand = randNSVec[currentWorkNodeNumber].getInstance();
		BoundaryStrategy * boundaryStrategy = BoundaryStrategy::getInstance();

		//iterating over subgridSections
		for (int s = 0; s < subgridSectionOrderVec.size(); ++s) {

			pair<Dim3D, Dim3D> sectionDims = pUtils->getPottsSection(currentWorkNodeNumber, s);
			//numberOfAttemptsLocal = (int)(sectionDims.second.x - sectionDims.first.x)*(sectionDims.second.y - sectionDims.first.y)*(sectionDims.second.z - sectionDims.first.z)*sim->getFlip2DimRatio();

			// #pragma omp critical


			for (unsigned int i = 0; i < numberOfAttempts; ++i) {



				//////currentAttempt=i;
				//currentAttemptLocal=i; //this will need to be fixed as this is used in PDE SOLVER CALLER
				//run fixed steppers - they are executed regardless whether spin flip take place or not . Note, regular stepers are executed only after spin flip attepmts takes place 

				if (fixedSteppers.size()) {
#pragma omp critical
					{
						//IMPORTANT: fixed steppers cause really bad performance with multiple processor runs. Currently only two of them are supported 
						// PDESolverCaller plugin and Secretion plugin. PDESolverCaller is deprecated and Secretion plugin section that mimics functionality of PDE sovler Secretion data section is depreecated as well
						// However Secretion plugin can be used to do "per cell" secretion (using python scripting). This will not slow down multicore simulation

						//cerr<<"pUtils->getCurrentWorkNodeNumber()="<<pUtils->getCurrentWorkNodeNumber()<<" currentAttempt="<<currentAttempt<<endl;
						for (unsigned int j = 0; j < fixedSteppers.size(); j++)
							fixedSteppers[j]->step();

						++currentAttempt; //to be consistent with serial code currentAttampt has to be increamented after fixedSteppers run 

					}
				}

				Point3D pt;

				//boundaryPointVector.assign(boundaryPixelSet.begin(), boundaryPixelSet.end());

				// Pick a random integer and pick a random point from a boundary
				pt = randomPickBoundaryPixel(rand);

				//////boundaryPointIndex = rand->getInteger(0, boundaryPixelSet.size() - 1);
				//////sitr = boundaryPixelSet.begin();
				//////
				//////advance(sitr, boundaryPointIndex);
				//////continue;
				//////pt = *sitr;

				//pt = boundaryPointVector[boundaryPointIndex];


				//boundaryPointVector.assign(boundaryPixelSet.begin(), boundaryPixelSet.end());

				//// Pick a random integer and pick a random point from a boundary
				//boundaryPointIndex = rand->getInteger(0, boundaryPointVector.size() - 1);
				//pt = boundaryPointVector[boundaryPointIndex];

				CellG *cell = cellFieldG->getQuick(pt);

				if (sizeFrozenTypeVec && cell) {///must also make sure that cell ptr is different 0; Will never freeze medium
					if (checkIfFrozen(cell->type))
						continue;
				}

				unsigned int directIdx = rand->getInteger(0, maxNeighborIndex);

				Neighbor n = boundaryStrategy->getNeighborDirect(pt, directIdx);

				if (!n.distance) {
					//if distance is 0 then the neighbor returned is invalid
					continue;
				}
				Point3D changePixel = n.pt;

				//check if changePixel refers to different cell. 
				CellG* changePixelCell = cellFieldG->getQuick(changePixel);

				if (changePixelCell == cell) {
					continue;//skip the rest of the loop if change pixel points to the same cell as pt
				}

				if (sizeFrozenTypeVec && changePixelCell) {///must also make sure that cell ptr is different 0; Will never freeze medium
					if (checkIfFrozen(changePixelCell->type))
						continue;
				}
				++attemptedECVec[currentWorkNodeNumber];

				flipNeighborVec[currentWorkNodeNumber] = pt;


				/// change takes place at change pixel  and pt is a neighbor of changePixel
				// Calculate change in energy
				double change = energyCalculator->changeEnergy(changePixel, cell, changePixelCell, i);

				// Acceptance based on probability
				double motility = fluctAmplFcn->fluctuationAmplitude(cell, changePixelCell);

				double prob = acceptanceFunction->accept(motility, change);


				if (numberOfThreads == 1) {
					energyCalculator->set_aceptance_probability(prob);
				}

				if (prob >= 1.0 || rand->getRatio() < prob) {

					// Accept the change
					energyVec[currentWorkNodeNumber] += change;

					if (connectivityConstraint && connectivityConstraint->changeEnergy(changePixel, cell, changePixelCell)) {
						//put call to durand fragmentation in this if
						// new if:

						//if (connectivityConstraint || durandContraint)
						//{ if(connectivityConstraint && connectivityConstraint->changeEnergy(changePixel, cell, changePixelCell))
						//  {...}
						//  if(durandContraint)
						//  {...}
						if (numberOfThreads == 1) {
							energyCalculator->setLastFlipAccepted(false);
						}
					}
					else {
						cellFieldG->set(changePixel, cell);
						flipsVec[currentWorkNodeNumber]++;
						if (numberOfThreads == 1) {
							energyCalculator->setLastFlipAccepted(true);
						}
					}
				}
				else {
					if (numberOfThreads == 1) {
						energyCalculator->setLastFlipAccepted(false);
					}
				}

				// Run steppers
				//#pragma omp single 
				{
					for (unsigned int j = 0; j < steppers.size(); j++)
						steppers[j]->step();
				}
			}

#pragma omp barrier
		}//iteration over subrid sections

#pragma omp critical
		{

			energy += energyVec[currentWorkNodeNumber];
			attemptedEC += attemptedECVec[currentWorkNodeNumber];
			flips += flipsVec[currentWorkNodeNumber];

			//reseting values before processing new slice

			energyVec[currentWorkNodeNumber] = 0.0;
			attemptedECVec[currentWorkNodeNumber] = 0;
			flipsVec[currentWorkNodeNumber] = 0;
		}

	} //pragma omp parallel

	if (debugOutputFrequency && !(currentStep % debugOutputFrequency)) {
		cerr << "Number of Attempted Energy Calculations=" << attemptedEC << endl;
	}
	return flips;

}

void Potts3D::setMetropolisAlgorithm(std::string _algName) {

	string algName = _algName;
	changeToLower(algName);

	if (algName == "list") {
		metropolisFcnPtr = &Potts3D::metropolisList;
	}
	else if (algName == "fast") {
		metropolisFcnPtr = &Potts3D::metropolisFast;
	}
	else if (algName == "boundarywalker") {
		metropolisFcnPtr = &Potts3D::metropolisBoundaryWalker;
	}
	else {
		metropolisFcnPtr = &Potts3D::metropolisFast;
	}

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Potts3D::setCellTypeMotilityVec(std::vector<float> & _cellTypeMotilityVec) {
	cellTypeMotilityVec = _cellTypeMotilityVec;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Potts3D::initializeCellTypeMotility(std::vector<CellTypeMotilityData> & _cellTypeMotilityVector) {

	ASSERT_OR_THROW("AUTOMATON IS NOT INITIALIZED", automaton);

	unsigned int typeIdMax = 0;
	//finding max typeId
	for (int i = 0; i < _cellTypeMotilityVector.size(); ++i) {
		//cerr<<"CHECKING "<<_cellTypeMotilityVector[i].typeName<<endl;
		unsigned int id = automaton->getTypeId(_cellTypeMotilityVector[i].typeName);
		if (id > typeIdMax)
			typeIdMax = id;
	}

	cellTypeMotilityVec.assign(typeIdMax + 1, 0.0);
	for (int i = 0; i < _cellTypeMotilityVector.size(); ++i) {
		unsigned int id = automaton->getTypeId(_cellTypeMotilityVector[i].typeName);
		cellTypeMotilityVec[id] = _cellTypeMotilityVector[i].motility;
	}

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool Potts3D::checkIfFrozen(unsigned char _type) {



	for (unsigned int i = 0; i < frozenTypeVec.size(); ++i) {
		if (frozenTypeVec[i] == _type)
			return true;
	}
	return false;

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Potts3D::setFrozenTypeVector(std::vector<unsigned char> & _frozenTypeVec) {
	frozenTypeVec = _frozenTypeVec;
	sizeFrozenTypeVec = frozenTypeVec.size();
}

void Potts3D::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

	bool fluctAmplGlobalReadFlag = false;
	if (_xmlData->getFirstElement("FluctuationAmplitude")) {
		if (!_xmlData->getFirstElement("FluctuationAmplitude")->findElement("FluctuationAmplitudeParameters")) {

			//we do not allow steering for motility specified by type
			sim->ppdCC3DPtr->temperature = _xmlData->getFirstElement("FluctuationAmplitude")->getDouble();
			fluctAmplGlobalReadFlag = true;
		}
	}


	if (!fluctAmplGlobalReadFlag && _xmlData->getFirstElement("Temperature")) {
		sim->ppdCC3DPtr->temperature = _xmlData->getFirstElement("Temperature")->getDouble();
	}



	if (_xmlData->getFirstElement("RandomSeed")) {
		BasicRandomNumberGenerator *rand = BasicRandomNumberGenerator::getInstance();
		if (rand->getSeed() != _xmlData->getFirstElement("RandomSeed")->getUInt())
			rand->setSeed(_xmlData->getFirstElement("RandomSeed")->getUInt());
	}

	if (_xmlData->getFirstElement("Offset")) {
		if (_xmlData->getFirstElement("Offset")->getDouble() != 0.)
			getAcceptanceFunction()->setOffset(_xmlData->getFirstElement("Offset")->getDouble());
	}
	if (_xmlData->getFirstElement("KBoltzman")) {
		if (_xmlData->getFirstElement("KBoltzman")->getDouble() != 1.0)
			getAcceptanceFunction()->setK(_xmlData->getFirstElement("KBoltzman")->getDouble());
	}
	if (_xmlData->getFirstElement("DebugOutputFrequency")) {
		if (debugOutputFrequency != _xmlData->getFirstElement("DebugOutputFrequency")->getUInt()) {
			setDebugOutputFrequency(_xmlData->getFirstElement("DebugOutputFrequency")->getUInt() > 0 ? _xmlData->getFirstElement("DebugOutputFrequency")->getUInt() : 0);
			sim->ppdCC3DPtr->debugOutputFrequency = debugOutputFrequency;
		}
	}

	bool depthFlag = false;
	unsigned int neighborOrder = 1; //safe to request as a default neighbororder 1. BoundaryStrategy will reinitialize neighbor list only if the new neighbor order is greater than the previous one
	float depth = 0.0;
	if (_xmlData->getFirstElement("FlipNeighborMaxDistance")) {

		depth = _xmlData->getFirstElement("FlipNeighborMaxDistance")->getDouble();
		depthFlag = true;
	}

	if (_xmlData->getFirstElement("NeighborOrder")) {

		neighborOrder = _xmlData->getFirstElement("NeighborOrder")->getUInt();
		depthFlag = false;
	}

	if (depthFlag) {
		setDepth(depth);
	}
	else {
		setNeighborOrder(neighborOrder);
	}

	//CustomAcceptanceFunction
	unsigned int currentStep = sim->getStep();
	if (_xmlData->getFirstElement("CustomAcceptanceFunction")) {
		if (currentStep > 0) {
			//we can only update custom acceptance function when simulation has been initialized and we know how many cores are used
			customAcceptanceFunction.update(_xmlData->getFirstElement("CustomAcceptanceFunction"), true);
		}

		//first  initialization of the acceptance function will be done in the metropolis function
		customAcceptanceExpressionDefined = true;
		customAcceptanceFunction.update(_xmlData->getFirstElement("CustomAcceptanceFunction"), false); //this stores XML information inside ExpressionEvaluationDepot local variables
		registerAcceptanceFunction(&customAcceptanceFunction);
	}



	//Units
	if (_xmlData->getFirstElement("Units")) {
		if (_xmlData->getFirstElement("Units")->findAttribute("DoNotDisplayUnits")) {
			displayUnitsFlag = false;
		}
		CC3DXMLElement *unitElemPtr;
		CC3DXMLElement *unitsPtr = _xmlData->getFirstElement("Units");
		unitElemPtr = unitsPtr->getFirstElement("MassUnit");
		if (unitElemPtr) {
			massUnit = Unit(unitElemPtr->getText());
		}
		unitElemPtr = unitsPtr->getFirstElement("LengthUnit");
		if (unitElemPtr) {
			lengthUnit = Unit(unitElemPtr->getText());
		}
		unitElemPtr = unitsPtr->getFirstElement("TimeUnit");
		if (unitElemPtr) {
			timeUnit = Unit(unitElemPtr->getText());
		}

		if (displayUnitsFlag) {
			updateUnits(unitsPtr);
		}

	}
	else {

		//displaying basic units

		CC3DXMLElement * unitsPtr = _xmlData->attachElement("Units", "");
		if (displayUnitsFlag) {
			updateUnits(unitsPtr);
		}

	}

}


void Potts3D::updateUnits(CC3DXMLElement * _unitsPtr) {


	////displaying basic units

	//if (_unitsPtr->getFirstElement("MassUnit")) {
	//	_unitsPtr->getFirstElement("MassUnit")->updateElementValue(massUnit.toString());
	//}
	//else {
	//	_unitsPtr->attachElement("MassUnit", massUnit.toString());
	//}

	//if (_unitsPtr->getFirstElement("LengthUnit")) {
	//	_unitsPtr->getFirstElement("LengthUnit")->updateElementValue(lengthUnit.toString());
	//}
	//else {
	//	_unitsPtr->attachElement("LengthUnit", lengthUnit.toString());
	//}

	//if (_unitsPtr->getFirstElement("TimeUnit")) {
	//	_unitsPtr->getFirstElement("TimeUnit")->updateElementValue(timeUnit.toString());
	//}
	//else {
	//	_unitsPtr->attachElement("TimeUnit", timeUnit.toString());
	//}

	//if (_unitsPtr->getFirstElement("VolumeUnit")) {
	//	_unitsPtr->getFirstElement("VolumeUnit")->updateElementValue(powerUnit(lengthUnit, 3).toString());
	//}
	//else {
	//	_unitsPtr->attachElement("VolumeUnit", powerUnit(lengthUnit, 3).toString());
	//}

	//if (_unitsPtr->getFirstElement("SurfaceUnit")) {
	//	_unitsPtr->getFirstElement("SurfaceUnit")->updateElementValue(powerUnit(lengthUnit, 2).toString());
	//}
	//else {
	//	_unitsPtr->attachElement("SurfaceUnit", powerUnit(lengthUnit, 2).toString());
	//}

	//energyUnit = massUnit*lengthUnit*lengthUnit / (timeUnit*timeUnit);

	//if (_unitsPtr->getFirstElement("EnergyUnit")) {
	//	_unitsPtr->getFirstElement("EnergyUnit")->updateElementValue(energyUnit.toString());
	//}
	//else {
	//	_unitsPtr->attachElement("EnergyUnit", energyUnit.toString());
	//}
}

std::string Potts3D::steerableName() {
	return "Potts";
}

