#include "CAManager.h"
#include "CACell.h"
#include "CACellFieldChangeWatcher.h"

#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Field3D/Point3D.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>

#include <iostream>
#include <sstream>
#include <string>


using namespace std;
using namespace CompuCell3D;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CAManager::CAManager():cellField(0),recentlyCreatedCellId(1),recentlyCreatedClusterId(1)
{}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CAManager::~CAManager()
{}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CAManager::createCellField(const Dim3D & _dim){

    cerr<<"creating field"<<endl;
    cerr<<"Will create field of dimension="<<_dim<<endl;
	RUNTIME_ASSERT_OR_THROW("CA: createCellField() cell field already created!", !cellField);
	cellField = new WatchableField3D<CACell *>(_dim, 0); //added    
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
void CAManager::registerCellFieldChangeWatcher(CACellFieldChangeWatcher * _watcher){
	RUNTIME_ASSERT_OR_THROW("registerCellFieldChangeWatcher _watcher cannot be NULL!",_watcher);
	RUNTIME_ASSERT_OR_THROW("registerCellFieldChangeWatcher cellField cannot be NULL!",cellField);

	cellField->addChangeWatcher(_watcher); 

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CAManager::destroyCell(CACell *  _cell, bool _flag){
	cerr<<"inside destroy cell"<<endl;
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
CACell * CAManager::createCell(long _clusterId){
	CACell * cell = new CACell();
	//////cell->extraAttribPtr=cellFactoryGroup.create();
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
	cerr<<"inventory size="<<cellInventory.getSize()<<endl;
	//////if(attrAdder){
	//////	attrAdder->addAttribute(cell);
	//////}
	return cell;


}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

