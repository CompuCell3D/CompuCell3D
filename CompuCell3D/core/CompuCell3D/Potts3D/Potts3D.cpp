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

// #include <CompuCell3D/Field3D/WatchableField3D.h>

#include <CompuCell3D/Potts3D/TypeTransition.h>
#include "EnergyFunctionCalculator.h"
#include "EnergyFunctionCalculatorStatistics.h"

#include <CompuCell3D/Simulator.h>
//#include <CompuCell3D/plugins/Volume/VolumePlugin.h>
//#include <CompuCell3D/plugins/Surface/SurfacePlugin.h>

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
		recentlyCreatedCellId = 1;
		recentlyCreatedClusterId = 1;
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
	cerr << "\t\t\t\t\t setDepth  maxNeighborIndex=" << maxNeighborIndex << endl;
	neighbors.clear();
	neighbors.assign(maxNeighborIndex + 1, Point3D());

}

void Potts3D::setNeighborOrder(unsigned int _neighborOrder) {
	BoundaryStrategy::getInstance()->prepareNeighborListsBasedOnNeighborOrder(_neighborOrder);
	maxNeighborIndex = BoundaryStrategy::getInstance()->getMaxNeighborIndexFromNeighborOrder(_neighborOrder);
	cerr << "\t\t\t\t\t setNeighborOrder  maxNeighborIndex=" << maxNeighborIndex << endl;
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

////////added for more convenient Python scripting
//////// notice that SWIG will automatically convert tupples and lists to vectors of a given type provided 
//////// they are passed by a constant reference
//////void Potts3D::resizeCellField(const vector<int> & dim, const vector<int> & shiftVec) {
//////	if (dim.size()==3 && shiftVec.size()==3){
//////		resizeCellField(Dim3D(dim[0],dim[1],dim[2]),Dim3D(shiftVec[0],shiftVec[1],shiftVec[2]));
//////	}
//////}


void Potts3D::registerAttributeAdder(AttributeAdder * _attrAdder) {
	attrAdder = _attrAdder;
}

// void Potts3D::registerTypeChangeWatcher(TypeChangeWatcher * _typeChangeWatcher){
//    typeTransition->registerTypeChangeWatcher(_typeChangeWatcher);
// }

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
	cell->id = recentlyCreatedCellId;
	++recentlyCreatedCellId;

	//this means that cells with clusterId<=0 should be placed at the end of PIF file if automatic numbering of clusters is to work for a mix of clustered and non clustered cells	

	if (_clusterId <= 0) { //default behavior if user does not specify cluster id or cluster id is 0

		cell->clusterId = recentlyCreatedClusterId;
		++recentlyCreatedClusterId;

	}
	else if (_clusterId > recentlyCreatedClusterId) { //clusterId specified by user is greater than recentlyCreatedClusterId

		cell->clusterId = _clusterId;
		recentlyCreatedClusterId = _clusterId + 1; // if we get cluster id greater than recentlyCreatedClusterId we set recentlyCreatedClusterId to be _clusterId+1
		// this way if users add "non-cluster" cells after definition of clustered cells	the cell->clusterId is guaranteed to be greater than any of the clusterIds specified for clustered cells
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
	cell->id = _cellId;

	if (_cellId >= recentlyCreatedCellId) {
		recentlyCreatedCellId = _cellId + 1;
	}


	//this means that cells with clusterId<=0 should be placed at the end of PIF file if automatic numbering of clusters is to work for a mix of clustered and non clustered cells	

	if (_clusterId <= 0) { //default behavior if user does not specify cluster id or cluster id is 0

		cell->clusterId = recentlyCreatedClusterId;
		++recentlyCreatedClusterId;

	}
	else if (_clusterId > recentlyCreatedClusterId) { //clusterId specified by user is greater than recentlyCreatedClusterId

		cell->clusterId = _clusterId;
		recentlyCreatedClusterId = _clusterId + 1; // if we get cluster id greater than recentlyCreatedClusterId we set recentlyCreatedClusterId to be _clusterId+1
		// this way if users add "non-cluster" cells after definition of clustered cells	the cell->clusterId is guaranteed to be greater than any of the clusterIds specified for clustered cells
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
	ASSERT_OR_THROW("Potts3D: cell field G not initialized", cellFieldG);

	// ParallelUtilsOpenMP * pUtils=sim->getParallelUtils();


	ASSERT_OR_THROW("MetropolisList algorithm works only in single processor mode. Please change number of processors to 1", pUtils->getNumberOfWorkNodesPotts() == 1);

	if (customAcceptanceExpressionDefined) {
		customAcceptanceFunction.initialize(this->sim); //actual initialization will happen only once at MCS=0 all other calls will return without doing anything
	}

	//here we will allocate Random number generators for each thread. Note that since user may change number of work nodes we have to monitor if the max number of work threads is greater than size of random number generator vector 
	if (!randNSVec.size() || pUtils->getMaxNumberOfWorkNodesPotts() > randNSVec.size()) { //each thread will have different random number ghenerator
		randNSVec.assign(pUtils->getMaxNumberOfWorkNodesPotts(), BasicRandomNumberGeneratorNonStatic());

		for (unsigned int i = 0; i <randNSVec.size(); ++i) {
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


			//     pt.x = rand->getInteger(0, dim.x - 1);
			//     pt.y = rand->getInteger(0, dim.y - 1);
			//     pt.z = rand->getInteger(0, dim.z - 1);

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

		for (unsigned int i = 0; i <randNSVec.size(); ++i) {
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

	//cerr<<"flipNeighborVec.size()="<<flipNeighborVec.size()<<endl;

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
	ASSERT_OR_THROW("Potts3D: You must supply an acceptance function!",
		acceptanceFunction);



	//FOR NOW WE WILL IGNORE BOX WATCHER FOR POTTS SECTION IT WILL STILL WORK WITH PDE SOLVERS
	Dim3D fieldDim = cellFieldG->getDim();
	numberOfAttempts = (int)fieldDim.x*fieldDim.y*fieldDim.z*sim->getFlip2DimRatio();
	unsigned int currentStep = sim->getStep();
	if (debugOutputFrequency && !(currentStep % debugOutputFrequency)) {
		cerr << "FAST numberOfAttempts=" << numberOfAttempts << endl;
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



				flipNeighborVec[currentWorkNodeNumber] = pt;/// change takes place at change pixel  and pt is a neighbor of changePixel
				// Calculate change in energy
				//cerr<<"steps="<<steps<<" temp="<<temp<<" acceptanceFunction="<<acceptanceFunction<<endl;

				double change = energyCalculator->changeEnergy(changePixel, cell, cellFieldG->get(changePixel), i);

				//cerr<<"This is change: "<<change<<endl;	

				//cerr<<"steps="<<steps<<" temp="<<temp<<" acceptanceFunction="<<acceptanceFunction<<endl;

				// Acceptance based on probability
				double motility = fluctAmplFcn->fluctuationAmplitude(cell, cellFieldG->get(changePixel));

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

					if (connectivityConstraint && connectivityConstraint->changeEnergy(changePixel, cell, cellFieldG->get(changePixel))) {
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

				//     exit(0);
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


unsigned int Potts3D::metropolisBoundaryWalker(const unsigned int steps, const double temp) {
	ASSERT_OR_THROW("Potts3D: cell field G not initialized", cellFieldG);

	// // // ParallelUtilsOpenMP * pUtils=sim->getParallelUtils();


	if (customAcceptanceExpressionDefined) {
		customAcceptanceFunction.initialize(this->sim); //actual initialization will happen only once at MCS=0 all other calls will return without doing anything
	}


	ASSERT_OR_THROW("BoundaryWalker Algorithm works only in single processor mode. Please change number of processors to 1", pUtils->getNumberOfWorkNodesPotts() == 1);

	//here we will allocate Random number generators for each thread. Note that since user may change number of work nodes we have to monitor if the max number of work threads is greater than size of random number generator vector 
	if (!randNSVec.size() || pUtils->getMaxNumberOfWorkNodesPotts() > randNSVec.size()) { //each thread will have different random number ghenerator
		randNSVec.assign(pUtils->getMaxNumberOfWorkNodesPotts(), BasicRandomNumberGeneratorNonStatic());

		for (unsigned int i = 0; i <randNSVec.size(); ++i) {
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
	//cerr<<"steps="<<steps<<" temp="<<temp<<" acceptanceFunction="<<acceptanceFunction<<endl;

	//numberOfAttempts=(int)(maxCoordinates.x-minCoordinates.x)*(maxCoordinates.y-minCoordinates.y)*(maxCoordinates.z-minCoordinates.z)*sim->getFlip2DimRatio();
	//   numberOfAttempts=steps;
	//Number of attempts is equal to number of boundary pixels
	numberOfAttempts = boundaryPixelSet.size();
	unsigned int currentStep = sim->getStep();
	if (debugOutputFrequency && !(currentStep % debugOutputFrequency)) {
		cerr << "numberOfAttempts=" << numberOfAttempts << endl;
	}

	long boundaryPointIndex;
	long counter = 0;
	set<Point3D>::iterator sitr;
	vector<Point3D> boundaryPointVector;
	boundaryPointVector.assign(boundaryPixelSet.begin(), boundaryPixelSet.end());

	BoundaryStrategy * boundaryStrategy = BoundaryStrategy::getInstance();
	//cerr<<"numberOf workNodes="<<pUtils->getNumberOfWorkNodesPotts()<<endl;
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

			// Pick a random integer
			boundaryPointIndex = rand->getInteger(minCoordinates.x, boundaryPointVector.size() - 1);
			pt = boundaryPointVector[boundaryPointIndex];

			//cerr<<"pt="<<pt<<" boundaryPointIndex="<<boundaryPointIndex<<endl;
			//    boundaryPointIndex=rand->getInteger(minCoordinates.x, boundaryPixelSet.size()-1);//we use most up-to-date set size
			//  counter=0;
			//for (set<Point3D>::iterator sitr=boundaryPixelSet.begin() ; sitr!=boundaryPixelSet.end(); ++sitr){
			// if(counter==boundaryPointIndex){
			//	pt=*sitr;
			//	break;
			// }
			// ++counter;
			//}

			//    cerr<<"pt flip="<<pt<<endl;
			//     pt.x = rand->getInteger(0, dim.x - 1);
			//     pt.y = rand->getInteger(0, dim.y - 1);
			//     pt.z = rand->getInteger(0, dim.z - 1);
			//     cerr<<"pt="<<pt<<" rand->getSeed()="<<rand->getSeed()<<endl;


			///Cell *cell = cellField->get(pt);
			CellG *cell = cellFieldG->get(pt);

			if (sizeFrozenTypeVec && cell) {///must also make sure that cell ptr is different 0; Will never freeze medium
				if (checkIfFrozen(cell->type))
					continue;
			}

			unsigned int directIdx = rand->getInteger(0, maxNeighborIndex);
			//cerr<<"directIdx="<<directIdx<<endl;

			Neighbor n = boundaryStrategy->getNeighborDirect(pt, directIdx);
			//cerr<<"n.pt"<<n.pt<<" n.distance="<<n.distance<<endl;

			if (!n.distance) {
				//if distance is 0 then the neighbor returned is invalid
				continue;
			}
			Point3D changePixel = n.pt;
			//       cerr<<"pt="<<pt<<" n.pt="<<n.pt<<" difference="<<pt-n.pt<<endl;
			//check if changePixel refers to different cell. 

			if (cellFieldG->getQuick(changePixel) == cell) {
				continue;//skip the rest of the loop if change pixel points to the same cell as pt
			}
			else {
				;

			}

			//cerr<<"pt="<<pt<<" changePixel="<<changePixel<<endl;

			if (sizeFrozenTypeVec && cellFieldG->get(changePixel)) {///must also make sure that cell ptr is different 0; Will never freeze medium
				if (checkIfFrozen(cellFieldG->get(changePixel)->type))
					continue;
			}
			++attemptedEC;

			flipNeighbor = pt;/// change takes place at change pixel  and pt is a neighbor of changePixel

			// Calculate change in energy
			//cerr<<"steps="<<steps<<" temp="<<temp<<" acceptanceFunction="<<acceptanceFunction<<endl;

			double change = energyCalculator->changeEnergy(changePixel, cell, cellFieldG->get(changePixel), i);

			//cerr<<"This is change: "<<change<<endl;	

			//cerr<<"steps="<<steps<<" temp="<<temp<<" acceptanceFunction="<<acceptanceFunction<<endl;

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
			//cerr<<"prob="<<prob<<endl;


			//cerr<<"change E="<<change<<" prob="<<prob<<endl;
			if (prob >= 1 || rand->getRatio() < prob) {
				// Accept the change
				//        cerr<<"accepted flip "<< change<<endl;
				energy += change;
				//       cerr<<"energy="<<energy<<endl;
				if (connectivityConstraint && connectivityConstraint->changeEnergy(changePixel, cell, cellFieldG->get(changePixel))) {
					energyCalculator->setLastFlipAccepted(false);
				}
				else {
					//cerr<<"FLIP ACCEPTED"<<endl;
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

			//     exit(0);
		}
	}// #pragma omp parallel
	if (debugOutputFrequency && !(currentStep % debugOutputFrequency)) {
		cerr << "Number of Attempted Energy Calculations=" << attemptedEC << endl;
	}
	//    exit(0);

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

	//for(int i =0 ; i < _cellTypeMotilityVector.size() ;++i ){
	//	
	//	cerr<<"cellTypeMotilityVec["<<i<<"]="<<cellTypeMotilityVec[i]<<endl;
	//}
	//
	//exit(0);
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
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//void Potts3D::update(ParseData *pd, bool _fullInitFlag){
//	PottsParseData * ppdPtr=(PottsParseData *)pd;
//	sim->ppdPtr=ppdPtr;
//	BasicRandomNumberGenerator *rand = BasicRandomNumberGenerator::getInstance();
//	if(rand->getSeed() != ppdPtr->seed)
//		rand->setSeed(ppdPtr->seed);
//
//	if(ppdPtr->offset!=0.){
//		getAcceptanceFunction()->setOffset(ppdPtr->offset);
//	}
//	if(ppdPtr->kBoltzman!=1.0){
//		getAcceptanceFunction()->setK(ppdPtr->kBoltzman);
//	}
//
//	if(debugOutputFrequency != ppdPtr->debugOutputFrequency){
//		setDebugOutputFrequency(ppdPtr->debugOutputFrequency >0 ?ppdPtr->debugOutputFrequency:1 );
//	}
//	if(ppdPtr->depthFlag){
//		setDepth(ppdPtr->depth);
//	}else{
//		setNeighborOrder(ppdPtr->neighborOrder);
//	}
//
//
//}


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


		//cerr<<"massUnit="<<massUnit.toString()<<endl;
		//cerr<<"lengthUnit="<<lengthUnit.toString()<<endl;
		//cerr<<"timeUnit="<<timeUnit.toString()<<endl;
		//cerr<<"updating units"<<endl;
		//exit(0);
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

		//CC3DXMLElement * massUnitElem = unitsPtr->attachElement("MassUnit",massUnit.toString());
		//
		//if( !massUnitElem  ){ //element alread exists
		//	unitsPtr->getFirstElement("MassUnit")->updateElementValue(massUnit.toString());
		//}

		//CC3DXMLElement * lengthUnitElem = unitsPtr->attachElement("LengthUnit",lengthUnit.toString());
		//
		//if( !lengthUnitElem  ){ //element alread exists
		//	unitsPtr->getFirstElement("LengthUnit")->updateElementValue(lengthUnit.toString());
		//}

		//CC3DXMLElement * timeUnitElem = unitsPtr->attachElement("TimeUnit",timeUnit.toString());
		//
		//if( !timeUnitElem  ){ //element alread exists
		//	unitsPtr->getFirstElement("TimeUnit")->updateElementValue(timeUnit.toString());
		//}

		//CC3DXMLElement * volumeUnitElem = unitsPtr->attachElement("VolumeUnit",powerUnit(lengthUnit,3).toString());
		//
		//if( !volumeUnitElem  ){ //element alread exists
		//	unitsPtr->getFirstElement("VolumeUnit")->updateElementValue(powerUnit(lengthUnit,3).toString());
		//}

		//CC3DXMLElement * surfaceUnitElem = unitsPtr->attachElement("SurfaceUnit",powerUnit(lengthUnit,2).toString());
		//
		//if( !surfaceUnitElem  ){ //element alread exists
		//	unitsPtr->getFirstElement("SurfaceUnit")->updateElementValue(powerUnit(lengthUnit,2).toString());
		//}


		//CC3DXMLElement * energyUnitElem = unitsPtr->attachElement("EnergyUnit",energyUnit.toString());
		//
		//if( !energyUnitElem  ){ //element alread exists
		//	unitsPtr->getFirstElement("EnergyUnit")->updateElementValue(energyUnit.toString());
		//}

	}

}


void Potts3D::updateUnits(CC3DXMLElement * _unitsPtr) {


	//displaying basic units

	if (_unitsPtr->getFirstElement("MassUnit")) {
		_unitsPtr->getFirstElement("MassUnit")->updateElementValue(massUnit.toString());
	}
	else {
		_unitsPtr->attachElement("MassUnit", massUnit.toString());
	}

	if (_unitsPtr->getFirstElement("LengthUnit")) {
		_unitsPtr->getFirstElement("LengthUnit")->updateElementValue(lengthUnit.toString());
	}
	else {
		_unitsPtr->attachElement("LengthUnit", lengthUnit.toString());
	}

	if (_unitsPtr->getFirstElement("TimeUnit")) {
		_unitsPtr->getFirstElement("TimeUnit")->updateElementValue(timeUnit.toString());
	}
	else {
		_unitsPtr->attachElement("TimeUnit", timeUnit.toString());
	}

	if (_unitsPtr->getFirstElement("VolumeUnit")) {
		_unitsPtr->getFirstElement("VolumeUnit")->updateElementValue(powerUnit(lengthUnit, 3).toString());
	}
	else {
		_unitsPtr->attachElement("VolumeUnit", powerUnit(lengthUnit, 3).toString());
	}

	if (_unitsPtr->getFirstElement("SurfaceUnit")) {
		_unitsPtr->getFirstElement("SurfaceUnit")->updateElementValue(powerUnit(lengthUnit, 2).toString());
	}
	else {
		_unitsPtr->attachElement("SurfaceUnit", powerUnit(lengthUnit, 2).toString());
	}

	energyUnit = massUnit*lengthUnit*lengthUnit / (timeUnit*timeUnit);

	if (_unitsPtr->getFirstElement("EnergyUnit")) {
		_unitsPtr->getFirstElement("EnergyUnit")->updateElementValue(energyUnit.toString());
	}
	else {
		_unitsPtr->attachElement("EnergyUnit", energyUnit.toString());
	}
}

std::string Potts3D::steerableName() {
	return "Potts";
}

