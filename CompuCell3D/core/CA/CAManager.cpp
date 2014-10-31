#include "CAManager.h"
#include "CACell.h"
#include "CACellStack.h"
#include "CACellFieldChangeWatcher.h"
#include "ProbabilityFunction.h"

#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Field3D/Point3D.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>


#include <BasicUtils/BasicRandomNumberGenerator.h>

#include <CompuCell3D/Boundary/BoundaryStrategy.h>

#include <CompuCell3D/Boundary/BoundaryTypeDefinitions.h>
#include <iostream>
#include <sstream>
#include <string>
#include <PublicUtilities/Algorithms_CC3D.h>

#include <algorithm>
//#define _DEBUG

using namespace std;
using namespace CompuCell3D;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CAManager::CAManager():
cellField(0),
cellFieldS(0),
recentlyCreatedCellId(1),
recentlyCreatedClusterId(1),
cellToDelete(0),
numSteps(0),
currentStep(0),
cellCarryingCapacity(1),
boundaryStrategy(0),
rand (0),
neighborOrder(1),
maxProb(1.0)
{

	//BoundaryStrategy::instantiate(ppdCC3DPtr->boundary_x, ppdCC3DPtr->boundary_y, ppdCC3DPtr->boundary_z, ppdCC3DPtr->shapeAlgorithm, ppdCC3DPtr->shapeIndex, ppdCC3DPtr->shapeSize, ppdCC3DPtr->shapeInputfile,HEXAGONAL_LATTICE);
	BoundaryStrategy::instantiate("noflux", "noflux", "noflux", "Default", 0, 0, "none",SQUARE_LATTICE);

	cellInventory.setCAManagerPtr(this);
	boundaryStrategy=BoundaryStrategy::getInstance();

	
	rand = new BasicRandomNumberGeneratorNonStatic;
	

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CAManager::registerProbabilityFunction(ProbabilityFunction * _fcn){
	probFcnRegistry.push_back(_fcn);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BoundaryStrategy * CAManager::getBoundaryStrategy(){
	return boundaryStrategy;
	//return BoundaryStrategy::getInstance();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CAManager::~CAManager()
{
	if (rand){
		delete rand;
		rand = 0;
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int CAManager::getCellCarryingCapacity(){
	return cellCarryingCapacity;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CAManager::setCellCarryingCapacity(int _cellCarryingCapacity){
	cellCarryingCapacity = _cellCarryingCapacity;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int CAManager::getNumSteps(){
	return numSteps;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CAManager::setNumSteps(int _numSteps){
	numSteps=_numSteps;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int CAManager::getCurrentStep(){
	return currentStep;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CAManager::step(int _i){
	currentStep=_i;
	runCAAlgorithm(_i);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CAManager::cleanAfterSimulation(){

	cellInventory.cleanInventory();

	//cleaning cellFieldS
	Dim3D fieldDim;
	fieldDim = cellFieldS->getDim();
	Point3D pt;
	for (pt.x=0 ; pt.x < fieldDim.x ; ++pt.x )
		for (pt.y=0 ; pt.y < fieldDim.y ; ++pt.y )
			for (pt.z=0 ; pt.z < fieldDim.z ; ++pt.z ){
				CACellStack * cellStack = cellFieldS->get(pt);
				if (cellStack){
					delete cellStack;
				}
			}
	return ;
}




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CAManager::createCellField(const Dim3D & _dim){
#ifdef _DEBUG
    cerr<<"creating field"<<endl;
    cerr<<"Will create field of dimension="<<_dim<<endl;
#endif
	RUNTIME_ASSERT_OR_THROW("CA: createCellField() cell field already created!", !cellField);
	cellField = new WatchableField3D<CACell *>(_dim, 0); //added    

	RUNTIME_ASSERT_OR_THROW("CA: createCellField() cell field S already created!", !cellFieldS);
	cellFieldS = new WatchableField3D<CACellStack *>(_dim, 0); //added    

	boundaryStrategy->setDim(_dim);

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CACellInventory * CAManager::getCellInventory(){
	return &cellInventory;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//WatchableField3D<CACell *> * CAManager::getCellField(){
//	return cellField;
//}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Field3D<CACell *> * CAManager::getCellField(){
	return cellField;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Field3D<CACellStack *> * CAManager::getCellFieldS(){
	return cellFieldS;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CAManager::registerCellFieldChangeWatcher(CACellFieldChangeWatcher * _watcher){
	RUNTIME_ASSERT_OR_THROW("registerCellFieldChangeWatcher _watcher cannot be NULL!",_watcher);
	RUNTIME_ASSERT_OR_THROW("registerCellFieldChangeWatcher cellField cannot be NULL!",cellField);

	cellField->addChangeWatcher(_watcher); 

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CAManager::setCellToDelete(CACell * _cell){ 
	//sets ptr of a cell to be deleted
	cellToDelete=_cell;

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CAManager::cleanup(){
	//used to delete cells
	if (cellToDelete){
		destroyCell(cellToDelete);
		cellToDelete=0;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CAManager::destroyCell(CACell *  _cell, bool _removeFromInventory){
#ifdef _DEBUG
	cerr<<"inside destroy cell"<<endl;
#endif

	if(_cell->extraAttribPtr){
		cellFactoryGroup.destroy(_cell->extraAttribPtr);
		_cell->extraAttribPtr=0;
	}
	if(_cell->pyAttrib && attrAdderPyObjectVec.size()){
        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();
		Py_DECREF(_cell->pyAttrib);
   
		PyGILState_Release(gstate);
	}

	//////if(_cell->pyAttrib && _attrAdder){
	//////	attrAdder->destroyAttribute(_cell);
	//////}

	//had to introduce these two cases because in the Cell inventory destructor we deallocate memory of pointers stored int the set
	//Since this is done during interation over set changing pointer (cell=0) or removing anything from set might corrupt container or invalidate iterators
	if(_removeFromInventory){
		cellInventory.removeFromInventory(_cell);
		delete _cell;
		_cell=0;
	}else{
		delete _cell;
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CAManager::positionCell(const Point3D &_pt,CACell *  _cell){
	if (!_cell) return; 

	//when we move cell to a different location, in CA we set its previous site to NULL ptr because CA cell can only occupy one lattice site 
	if (_cell->xCOM >= 0 ){
		//when cell has been initialized and allocated at least c to a lattice site (_cell->xCOM  is > 0) that when we move cell to a different location we have to assign old cell's site to NULL pointer
		cellField->set(Point3D(_cell->xCOM,_cell->yCOM,_cell->zCOM),0); 
	}

	cellField->set(_pt,_cell); 
	cleanup(); // in CA each repositionning of cell has high chance of overwritting (hence deleting) another cell. Therefore we call cleanup often

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CACell * CAManager::createAndPositionCell(const Point3D & pt, long _clusterId){
	
	RUNTIME_ASSERT_OR_THROW("CA: createAndPositionCell cell field needs to be created first!", cellField);
	RUNTIME_ASSERT_OR_THROW("createCell() cellField point out of range!", cellField->isValid(pt));
	
	CACell * cell=createCell(_clusterId);
	positionCell(pt, cell);

	//cellField->set(pt, cell);
	return cell;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CAManager::positionCellS(const Point3D &_pt,CACell *  _cell){
	if (!_cell) return; 
	CACellStack * cellStack = cellFieldS->get(_pt);
	
#ifdef _DEBUG
	cerr<<"cellStack ="<<cellStack <<endl;
#endif

	if (_cell->xCOM >= 0){
		//here we remove cell from its current location in the field of cell stacks
		CACellStack * currentStack = cellFieldS->get(Point3D(_cell->xCOM,_cell->yCOM,_cell->zCOM));		
		currentStack->deleteCell(_cell);
	}

	//here we assign cell to new cell stack
	if (! cellStack ){
		cellStack = new CACellStack(cellCarryingCapacity);
		cellFieldS->set(_pt,cellStack);
	}


	_cell->xCOM=_pt.x;
	_cell->yCOM=_pt.y;
	_cell->zCOM=_pt.z;

	cellToDelete = cellStack ->appendCellForce(_cell);

	if (cellToDelete){
		cerr<<" will be deleteing cell = "<<cellToDelete->id<<" cellToDelete="<<cellToDelete<<"  at location="<<_pt<<endl;
	}
#ifdef _DEBUG
	cerr<<"from appendCelLForce cellToDelete = "<<cellToDelete <<endl;
#endif
	cleanup(); // in CA each repositionning of cell has high chance of overwritting (hence deleting) another cell. Therefore we call cleanup often
	

	////when we move cell to a different location, in CA we set its previous site to NULL ptr because CA cell can only occupy one lattice site 
	//if (_cell->xCOM >= 0 ){
	//	//when cell has been initialized and allocated at least c to a lattice site (_cell->xCOM  is > 0) that when we move cell to a different location we have to assign old cell's site to NULL pointer
	//	cellField->set(Point3D(_cell->xCOM,_cell->yCOM,_cell->zCOM),0); 
	//}

	//cellField->set(_pt,_cell); 
	//cleanup(); // in CA each repositionning of cell has high chance of overwritting (hence deleting) another cell. Therefore we call cleanup often

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CACell * CAManager::createAndPositionCellS(const Point3D & pt, long _clusterId){
	
	RUNTIME_ASSERT_OR_THROW("CA: createAndPositionCell cell field needs to be created first!", cellFieldS);
	RUNTIME_ASSERT_OR_THROW("createCell() cellField point out of range!", cellFieldS->isValid(pt));
	
	CACell * cell=createCell(_clusterId);
	positionCellS(pt, cell);

	//cellField->set(pt, cell);
	return cell;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CAManager::registerClassAccessor(BasicClassAccessorBase *_accessor){
	RUNTIME_ASSERT_OR_THROW("registerClassAccessor() _accessor cannot be NULL!", _accessor);

	cellFactoryGroup.registerClass(_accessor);

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CAManager::registerPythonAttributeAdderObject(PyObject *_attrAdder){
	attrAdderPyObjectVec.push_back(_attrAdder);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CACell * CAManager::createCell(long _clusterId){
	CACell * cell = new CACell();
	cell->extraAttribPtr=cellFactoryGroup.create();

	cell->id=recentlyCreatedCellId;
	cell->clusterId=cell->id; // for now we do not allow compartmentalized cells
	++recentlyCreatedCellId;
	recentlyCreatedClusterId=recentlyCreatedCellId;

	////////this means that cells with clusterId<=0 should be placed at the end of PIF file if automatic numbering of clusters is to work for a mix of clustered and non clustered cells	

	//////if (_clusterId <= 0){ //default behavior if user does not specify cluster id or cluster id is 0

	//////	cell->clusterId=recentlyCreatedClusterId;
	//////	++recentlyCreatedClusterId;

	//////}else if(_clusterId > recentlyCreatedClusterId){ //clusterId specified by user is greater than recentlyCreatedClusterId

	//////	cell->clusterId=_clusterId;
	//////	recentlyCreatedClusterId=_clusterId+1; // if we get cluster id greater than recentlyCreatedClusterId we set recentlyCreatedClusterId to be _clusterId+1
	//////	// this way if users add "non-cluster" cells after definition of clustered cells	the cell->clusterId is guaranteed to be greater than any of the clusterIds specified for clustered cells
	//////}else{ // cluster id is greater than zero but smaller than recentlyCreatedClusterId
	//////	cell->clusterId=_clusterId;
	//////}


	cellInventory.addToInventory(cell);
#ifdef _DEBUG
	cerr<<"inventory size="<<cellInventory.getSize()<<endl;
#endif _DEBUG

	if (attrAdderPyObjectVec.size()){
		// since we are using threads (swig generated modules are fully "threaded") and and use C/API we better make sure that before doing anything in python
		//we aquire GIL and then release it once we are done
		PyGILState_STATE gstate;
		gstate = PyGILState_Ensure();
	
	   PyObject *obj;
	   obj = PyObject_CallMethod(attrAdderPyObjectVec[0],"addAttribute",0);
	   cell->pyAttrib=obj;

		PyGILState_Release(gstate);


	}
	//////if(attrAdder){
	//////	attrAdder->addAttribute(cell);
	//////}
	return cell;


}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CAManager::setNeighborOrder(int _no){
	neighborOrder=_no;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int CAManager::getNeighborOrder(){
	return neighborOrder;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CAManager::setMaxProb(float _maxProb){
	maxProb=_maxProb;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float CAManager::getMaxProb(){
	return maxProb;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CAManager::registerConcentrationField(std::string _name,Field3D<float>* _fieldPtr){
	std::map<std::string,Field3D<float>*>::iterator mitr = 	concentrationFieldNameMap.find(_name);
	RUNTIME_ASSERT_OR_THROW("Cannot register field. Field "+_name+" already registered", mitr==concentrationFieldNameMap.end());

	concentrationFieldNameMap.insert(std::make_pair(_name,_fieldPtr));
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::map<std::string,Field3D<float>*> & CAManager::getConcentrationFieldNameMap(){
	return concentrationFieldNameMap;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Field3D<float>* CAManager::getConcentrationField(std::string _fieldName){
	      std::map<std::string,Field3D<float>*> & fieldMap=this->getConcentrationFieldNameMap();
	  //cerr<<" mapSize="<<fieldMap.size()<<endl;
      std::map<std::string,Field3D<float>*>::iterator mitr;
      mitr=fieldMap.find(_fieldName);
      if(mitr!=fieldMap.end()){
         return mitr->second;
      }else{
         return 0;
      }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::string> CAManager::getConcentrationFieldNameVector(){
	vector<string> fieldNameVec;
	std::map<std::string,Field3D<float>*>::iterator mitr;
	for (mitr=concentrationFieldNameMap.begin()  ; mitr !=concentrationFieldNameMap.end() ; ++mitr){
		fieldNameVec.push_back(mitr->first);	
	}
	return fieldNameVec;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CAManager::runCAAlgorithm(int _mcs){

	Point3D pt;
	Dim3D fieldDim=cellFieldS->getDim();
	unsigned int maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(neighborOrder);

	long numberOfAttempts = fieldDim.x*fieldDim.y*fieldDim.z;

	cerr<<"maxNeighborIndex="<<maxNeighborIndex<<endl;
	vector<float> zeroVec;
	vector<float> probVec;
	vector<int> positionVec;
	vector<Point3D> neighborVec;


	zeroVec.assign(maxNeighborIndex+1,0.0);
	positionVec.assign(maxNeighborIndex+1,0);
	neighborVec.assign(maxNeighborIndex+1,Point3D());

	for (std::size_t i = 0 ; i < positionVec.size() ; ++i){
		positionVec[i]=i;
	}


	vector<CACell *> * cellVectorPtr = cellInventory.generateCellVector();
	vector<CACell *> & cellVector = *cellVectorPtr;

	long int cellVecSize = cellVector.size();

	////c++ is a mess when it comes to backward compatibiiuty random-shuffle algorithm is best example... decided to explicitely code it
	// for (long int i=cellVecSize-1; i>0; --i) {
	//	swap (cellVector[i],cellVector[rand->getInteger(0, i+1)]);
	// }

	shuffleSTLContainer(cellVector , *rand);

	cerr<<"cellVecSize ="<<cellVecSize <<endl;
	 for (long int i= 0 ; i < cellVecSize ; ++i){
		 CACell * cell = cellVector[i];
			 
		 probVec = zeroVec;
		 //cerr<<"cell="<<cell<<" type="<<(int)cell->type<<" id="<<cell->id<<endl;
		 pt.x = cell->xCOM;
		 pt.y = cell->yCOM;
		 pt.z = cell->zCOM;
		 //cerr<<"comPt="<<pt<<endl;

		 for (int idx = 0 ; idx <=maxNeighborIndex ; ++idx){
			//cerr<<"idx="<<idx<<endl;
			//cerr<<"boundaryStrategy="<<boundaryStrategy<<endl;
			Neighbor n = boundaryStrategy->getNeighborDirect(pt,idx);
			
			if(!n.distance) {
				neighborVec[idx]=Point3D();
				continue;
			}
			 
			double prob = 0.0;
			for (unsigned int pdx = 0 ; pdx < probFcnRegistry.size() ; ++pdx ){			
				prob += probFcnRegistry[pdx]->calculate(pt , n.pt);
			}
			
			//cerr<<"idx="<<idx<<" prob="<<prob<<endl;			
			 probVec[idx] = prob;
			 neighborVec[idx] = n.pt;
		 }
		 
		 //shuffling position vec that we will use in calculating probabilities
		 float randomNumber=rand->getRatio();
		 
		 float probTest=0.0;

		 Point3D changePixel ;

		 bool moveCell=false;
		 
		 shuffleSTLContainer(positionVec , *rand);
		 
		 //int zeroIdx=0;
		 //float beforeProb=0.0;
		 //float afterProb=0.0;
		 for (int idx = 0 ; idx < positionVec.size() ; ++idx){
			 //beforeProb=probTest;
			 probTest += probVec[positionVec[idx]];
			 //afterProb = probTest;
			 //cerr<<" idx="<<idx<<" positionVec[idx]="<<positionVec[idx]<<" probVec[positionVec[idx]]="<<probVec[positionVec[idx]]<<endl;
			 //cerr<<"neighborVec [positionVec[idx]]="<<neighborVec [positionVec[idx]]<<endl;
			 //if (probVec[positionVec[idx]]==0.0){
				// zeroIdx=idx;
				// cerr<<" idx="<<idx<<" positionVec[idx]="<<positionVec[idx]<<" probVec[positionVec[idx]]="<<probVec[positionVec[idx]]<<endl;
				// cerr<<" prob=0 neighborVec [positionVec[idx]]="<<neighborVec [positionVec[idx]]<<endl;
			 //}

			 if (randomNumber < probTest ){
				 //if (idx==zeroIdx){
					// cerr<<"randomNumber "<<randomNumber <<"probTest"<<probTest<<endl;
					// cerr<<"FINAL IDX="<<idx<<"  neighborVec [positionVec[idx]]="<<neighborVec [positionVec[idx]]<<endl;
					// cerr<<"beforeProb="<<beforeProb<<" afterProb="<<afterProb<<" check="<<(beforeProb==afterProb)<<endl;
				 //}
				 changePixel =neighborVec [positionVec[idx]];
				 moveCell=true;
				 break;
			 }

		 }
		 
		 //cerr<<"\n\n\n\nmoveCell="<<moveCell<<endl;
		 //cerr<<"changePixel="<<changePixel<<endl;
		 
		 
		 if (moveCell){

			//CACellStack * cellStack = cellFieldS->get(changePixel);
	


			////if (cell->xCOM >= 0){
			////	//here we remove cell from its current location in the field of cell stacks
			////	CACellStack * currentStack = cellFieldS->get(Point3D(cell->xCOM,cell->yCOM,cell->zCOM));		
			////	currentStack->deleteCell(cell);
			////}

			////here we assign cell to new cell stack
			//if (! cellStack ){
			//	cellStack = new CACellStack(cellCarryingCapacity);
			//	cellFieldS->set(changePixel,cellStack);
			//}

			////cerr<<"changePixel="<<changePixel<<endl;
			//cell->xCOM=changePixel.x;
			//cell->yCOM=changePixel.y;
			//cell->zCOM=changePixel.z;

			////cellToDelete = cellStack ->appendCellForce(cell);
			 
			 positionCellS(changePixel,cell);
		 }




	 }
	

	//for (long i =0; i < numberOfAttempts ; ++i){
	//	//pick source
	//	pt.x = rand->getInteger(0, fieldDim.x-1);
	//	pt.y = rand->getInteger(0, fieldDim.y-1);
	//	pt.z = rand->getInteger(0, fieldDim.z-1);
	//	
	//	unsigned int directIdx = rand->getInteger(0, maxNeighborIndex);
	//	Neighbor n = boundaryStrategy->getNeighborDirect(pt,directIdx);

	//	CACellStack * sourceStack = cellFieldS->get(pt);
	//	if (!sourceStack){
	//		continue; //no source stack
	//	}else if (sourceStack->getFillLevel()==0){
	//		continue ; //empty source stack
	//	}


	//	if(!n.distance){
	//		//if distance is 0 then the neighbor returned is invalid
	//		continue;
	//	}

	//	//targetPixel
	//	Point3D changePixel = n.pt;

	//	double prob = 0.0;
	//	for (unsigned int idx = 0 ; idx < probFcnRegistry.size() ; ++idx ){
	//		
	//		prob += probFcnRegistry[idx]->calculate(pt,changePixel);

	//	}



	//	if (rand->getRatio() < prob){
	//		//do the move

	//	}
	//	
	//	//cerr<<"prob="<<prob<<endl;

	//}
}