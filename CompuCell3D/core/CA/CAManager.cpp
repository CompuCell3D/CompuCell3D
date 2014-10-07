#include "CAManager.h"
#include "CACell.h"
#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Field3D/Point3D.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <iostream>


using namespace std;
using namespace CompuCell3D;

CAManager::CAManager():cellField(0),recentlyCreatedCellId(1),recentlyCreatedClusterId(1)
{}

CAManager::~CAManager()
{}

void CAManager::createCellField(const Dim3D & _dim){

    cerr<<"creating field"<<endl;
    cerr<<"Will create field of dimension="<<_dim<<endl;
	ASSERT_OR_THROW("CA: createCellField() cell field already created!", !cellField);
	cellField = new WatchableField3D<CACell *>(_dim, 0); //added    
}


void CAManager::destroyCell(CACell *  _cell, bool _flag){
	cerr<<"insode destroy cell"<<endl;
}

CACell * CAManager::createAndPositionCell(const Point3D & pt, long _clusterId){
	ASSERT_OR_THROW("CA: createAndPositionCell cell field needs to be created first!", cellField);
	ASSERT_OR_THROW("createCell() cellField point out of range!", cellField->isValid(pt));

	CACell * cell=createCell(_clusterId);

	cellField->set(pt, cell);

	return cell;

	
}

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

	//////if(attrAdder){
	//////	attrAdder->addAttribute(cell);
	//////}
	return cell;


}
