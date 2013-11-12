#include "Cell.h"
#include <iostream>


#include "CellInventory.h"
#include "Potts3D.h"
#include <limits>

#undef max
#undef min

using namespace std;

namespace CompuCell3D {




	CC3DCellList::~CC3DCellList(){}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	CellInventory::CellInventory():potts(0)
	{
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	void CellInventory::setPotts3DPtr(Potts3D *_potts){
		potts=_potts;
		compartmentInventory.setPotts3DPtr(_potts);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	CellInventory::~CellInventory()
	{	
		//using namespace std;
		////Freeing up cell inventory has to be done
		//CellInventory::cellInventoryIterator cInvItr;

		//CellG * cell;

		/////loop over all the cells in the inventory   
		//for( cInvItr = cellInventoryBegin() ; cInvItr !=cellInventoryEnd() ; ++cInvItr ){
		//	cell=getCell(cInvItr);
		//	//cell=*cInvItr;
		//	if(!potts){
		//		delete cell;
  //       }
  //       else{
		//		potts->destroyCellG(cell,false);
  //       }
		//}
		cleanInventory();
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	void CellInventory::cleanInventory(){
		using namespace std;
		//Freeing up cell inventory has to be done
		CellInventory::cellInventoryIterator cInvItr;

		CellG * cell;

		///loop over all the cells in the inventory   
		for( cInvItr = cellInventoryBegin() ; cInvItr !=cellInventoryEnd() ; ++cInvItr ){
			cell=getCell(cInvItr);
			//cell=*cInvItr;
			if(!potts){
				delete cell;
         }
         else{
				potts->destroyCellG(cell,false);
         }
		}
		inventory.clear();

	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	void CellInventory::addToInventory(CellG * _cell){
		inventory.insert(make_pair(CellIdentifier(_cell->id,_cell->clusterId),_cell));
		compartmentInventory.addToInventory(_cell);
		//inventory.insert(_cell);
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	void CellInventory::removeFromInventory(CellG * _cell){
		inventory.erase(CellIdentifier(_cell->id,_cell->clusterId));
		compartmentInventory.removeFromInventory(_cell);
		//cerr<<"ERASING CELL WITH ID="<<_cell->id<<" clusterID="<<_cell->clusterId<<endl;
		//inventory.erase(_cell);

	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// when changing  cluster id one needs to reposition the cell in the inventory - erasing previous entry for a given cell 
	bool CellInventory::reassignClusterId(CellG * _cell, long _newClusterId){
		cellInventoryIterator mitr = inventory.find(CellIdentifier(_cell->id,_newClusterId));
		if(mitr==inventory.end()){ //entry with cell->id,_newClusterId does not exist
			//////inventory.erase(CellIdentifier(_cell->id,_cell->clusterId));
			removeFromInventory(_cell);
			_cell->clusterId=_newClusterId;
			addToInventory(_cell);
			//cerr<<"inserted cell with id = "<<_cell->id<<" clusterId="<<_cell->clusterId<<endl;
			return true;

		}else{//entry with cell->id,_newClusterId exists
			return false; //this entry exist and one cannot reassign anything
		}


	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//This is linear search unfortunately. Need to restructure inventory to take advantage of binary search . Set is not appropriate structure for this
	//CellInventory::cellInventoryIterator CellInventory::find(long _id){
	//	CellInventory::cellInventoryIterator cInvItr;

	//	CellG * cell;

	//	///loop over all the cells in the inventory   
	//	for( cInvItr = cellInventoryBegin() ; cInvItr !=cellInventoryEnd() ; ++cInvItr ){
	//		cell=*cInvItr;
	//		if(cell->id==_id)
	//			return cInvItr;
	//	}
	//	return cellInventoryEnd();
	//}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//linear searh but this function will not be used too often. If necesary we can easily solve this problem later
	CellInventory::cellInventoryIterator CellInventory::find(CellG * _cell){
		CellInventory::cellInventoryIterator cInvItr;
		CellG * cell;
		for( cInvItr = cellInventoryBegin() ; cInvItr !=cellInventoryEnd() ; ++cInvItr ){
			cell=getCell(cInvItr);
			if (cell==_cell)
				return cInvItr;
		}
		return cellInventoryEnd();
	}


	CellG * CellInventory::getCellById(long _id){
		cellInventoryIterator cInvItr;
		cInvItr=inventory.find(_id);
			if(cInvItr!=inventory.end()){
				return cInvItr->second;
			}else{
				return 0;
			}
	}

	CellG * CellInventory::getCellByIds(long _id,long _clusterId){
		CellIdentifier cellId(_id,_clusterId);
		cellInventoryIterator cInvItr;
		cInvItr=inventory.find(cellId);
		
		if(cInvItr!=inventory.end()){
			return cInvItr->second;
		}else{
			return 0;
		}

	}

CellG * CellInventory::attemptFetchingCellById(long _id){

	cellInventoryIterator upperMitr=inventory.upper_bound(CellIdentifier(_id,std::numeric_limits<long>::max())); //upperMitr will point to location whose key is 'greater' than searched key
	if (upperMitr!=inventory.begin()){
		--upperMitr;
	}
	//cerr<<"trying to get id="<<_id<<endl;
	//cerr<<"upperMitr->first.cellId="<<upperMitr->first.cellId<<endl;

	if (upperMitr->first.cellId==_id){
		return upperMitr->second;
	}else{
		return 0;
		//--upperMitr;
		//if (upperMitr->first.cellId==_id){
		//	return upperMitr->second;
		//}else{
		//	return 0;
		//}

	}

	

	}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CC3DCellList CellInventory::getClusterCells(long _clusterId){
	return compartmentInventory.getClusterCells(_clusterId);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	void CellInventory::initCellInventoryByType(CellInventory::cellListByType_t *_inventoryByTypePtr,unsigned char _type){
		_inventoryByTypePtr->clear();
		CellInventory::cellInventoryIterator cInvItr;
		CellG * cell;
		for( cInvItr = cellInventoryBegin() ; cInvItr !=cellInventoryEnd() ; ++cInvItr ){
			cell=getCell(cInvItr);
			if (cell->type==_type)
				_inventoryByTypePtr->insert(make_pair(cell->id,cell));
				
		}			
	}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	void CellInventory::initCellInventoryByMultiType(CellInventory::cellListByType_t *_inventoryByTypePtr,std::vector<int> * _typeVecPtr){
		_inventoryByTypePtr->clear();
		CellInventory::cellInventoryIterator cInvItr;
		CellG * cell;
		vector<int> & typeVec=*_typeVecPtr;
		for( cInvItr = cellInventoryBegin() ; cInvItr !=cellInventoryEnd() ; ++cInvItr ){
			cell=getCell(cInvItr);
			for (unsigned int i = 0 ; i < typeVec.size(); ++i){
				if (cell->type==typeVec[i]){
					_inventoryByTypePtr->insert(make_pair(cell->id,cell));
					break;
				}


			}
				
		}			
	}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	


	CompartmentInventory::CompartmentInventory():
	potts(0)
	{
		
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	CompartmentInventory::~CompartmentInventory()	
	{		
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	void CompartmentInventory::setPotts3DPtr(Potts3D *_potts){
		potts=_potts;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	void CompartmentInventory::addToInventory(CellG * _cell){
		if (!_cell)
			return;
		compartmentInventoryIterator mitr=inventory.find(_cell->clusterId); //see if we have this clusterId in inventory already
		if(mitr!=inventory.end()){//cluster already exists
			compartmentListIterator msitr=mitr->second.find(_cell->id);//see if for this cluster id list of cells contains current cell id
			if (msitr!=mitr->second.end()){ //cell of this id already belongs to a cluster - no need to reinsert it
				return;
			}else{//have to insert new cell id to the list of cells of the cluster
				mitr->second.insert(make_pair(_cell->id,_cell));
			}

		}else{ //inserting new entry for new cluster
			compartmentListContainerType singleCellCluster;
			singleCellCluster.insert(make_pair(_cell->id,_cell));
			inventory.insert(make_pair(_cell->clusterId,singleCellCluster));
		}
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	void CompartmentInventory::removeClusterFromInventory(long _clusterId){

		inventory.erase(inventory.find(_clusterId)); 
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	void CompartmentInventory::removeFromInventory(CellG * _cell){
		if (!_cell)
			return;
		compartmentInventoryIterator mitr=inventory.find(_cell->clusterId); //see if we have this clusterId in inventory already
		if(mitr!=inventory.end()){//cluster already exists
			compartmentListIterator msitr=mitr->second.find(_cell->id);//see if for this cluster id list of cells contains current cell id
			if (msitr!=mitr->second.end()){ //cell of this id  belongs to a cluster - will erase it 
				mitr->second.erase(_cell->id);
				if(!mitr->second.size()){//if there is no more cells in the cluster remove the reference to cluster too
					inventory.erase(mitr);
				}
			}
		}		
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//bool CompartmentInventory::reassignClusterId(CellG *_cell,long _newClusterId){
	//	if (!_cell)
	//		return false;
	//	compartmentInventoryIterator mitr=inventory.find(_newClusterId);
	//	//see if cell with _cell->id and clusterId==_newClusterId already exists in the inventory if so no need to do any reassignments
	//	if(mitr!=inventory.end()){
	//		compartmentListIterator msitr=mitr->second.find(_cell->id);
	//		if(msitr!=mitr->second.end())
	//			return false;
	//	}
	//	removeCellFromInventory(_cell);
	//	_cell->clusterId=_newClusterId;
	//	addToInventory(_cell);
	//	return true;


	//}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	CC3DCellList CompartmentInventory::getClusterCells(long _clusterId){
		using namespace std;
		compartmentInventoryIterator mitr=inventory.find(_clusterId);
		if(mitr!=inventory.end()){
			CC3DCellList _cellVec;
			for(compartmentListIterator msitr=mitr->second.begin() ; msitr!=mitr->second.end() ; ++msitr){
				_cellVec.push_back(msitr->second);
			}
			return _cellVec;
		}
		return CC3DCellList ();
	}
};
