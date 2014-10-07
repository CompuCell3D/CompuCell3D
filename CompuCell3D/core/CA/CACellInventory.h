#ifndef CACELLINVENTORY_H
#define CACELLINVENTORY_H

#include <set>
#include <vector>
#include <map>
#include "CADLLSpecifier.h"
//NOTE: compartment inventory should be changed to cluster inventory to avoid name confusion

//#include <CompuCell3D/dllDeclarationSpecifier.h>

namespace CompuCell3D {

class CACell;
class CAManager;
/**
@author m
*/

class CASHARED_EXPORT CACellList:public std::vector<CACell*>{
//may add interface later if necessary
	public:
	typedef std::vector<CACell*>::iterator CACellListIterator_t;
	virtual ~CACellList();
	//added it to make python interfacing a bit easier - should implement separate interface in C++ anyway
	virtual std::vector<CACell*> * getBaseClass(){return (std::vector<CACell*> *)this;}
	


};

class CASHARED_EXPORT CACellIdentifier{
	public:
    
		CACellIdentifier(long _cellId=0):cellId(_cellId){}
		long cellId;
		
         ///have to define < operator if using a class in the set and no < operator is defined for this class
         bool operator<(const CACellIdentifier & _rhs) const{
			return cellId < _rhs.cellId ;

         }

};

class CASHARED_EXPORT CACellInventory
{
    public:
        typedef  std::map<CACellIdentifier,CACell *> cellInventoryContainerType;
        //typedef  std::set<CellG *> cellInventoryContainerType;
        typedef  cellInventoryContainerType::iterator cellInventoryIterator;
        typedef std::map<long,CACell *> cellListByType_t;


        CACellInventory();
        virtual ~CACellInventory();
        virtual void addToInventory(CACell * _cell);
        virtual void removeFromInventory(CACell * _cell);
        //std::set<CellG *>::size_type getCellInventorySize(){return inventory.size();}
        std::map<long,CACell *>::size_type getCellInventorySize(){return inventory.size();}
        int getSize(){return inventory.size();}   
        cellInventoryIterator cellInventoryBegin(){return inventory.begin();}
        cellInventoryIterator cellInventoryEnd(){return inventory.end();}
        void incrementIterator(cellInventoryIterator & _itr){++_itr;}
        void decrementIterator(cellInventoryIterator & _itr){--_itr;}
        bool reassignClusterId(CACell *,long);
        cellInventoryIterator find(CACell * _cell);
        cellInventoryIterator find(long _id){return inventory.find(_id);};
        cellInventoryContainerType & getContainer(){return inventory;}
		void setCAManagerPtr(CAManager *_caManager);
        CACell * getCellById(long _id); //obsolete
		CACell * getCellByIds(long _id,long clusterId);    // this implementation will ignore clusterId, for now     
		CACell * attemptFetchingCellById(long _id);


		CACell * getCell(cellInventoryIterator & _itr){return _itr->second;}
		// CACellList  getClusterCells(long _clusterId);

		void initCellInventoryByType(cellListByType_t *_inventoryByTypePtr,unsigned char _type); //the return variable is the same as the second argument 																											  
	    void initCellInventoryByMultiType(cellListByType_t *_inventoryByTypePtr,std::vector<int> * _typeVecPtr);

		// CompartmentInventory & getClusterInventory(){return compartmentInventory;}
		void cleanInventory();

      //BasicClassGroup * getPtr(cellInventoryIterator _itr){return const_cast<BasicClassGroup*>(*_itr); }
    private:
        cellInventoryContainerType inventory;
        CAManager *caManager;
      // CompartmentInventory compartmentInventory;
     
};

// // // //this inventory will have to be rewritten using less ad-hoc coding styles....

// // // class CompartmentInventory{

// // // public:
	// // // typedef std::map<long,CellG *> compartmentListContainerType;
	// // // typedef std::map<long,CellG *>::iterator compartmentListIterator;
	// // // typedef  std::map<long ,compartmentListContainerType > compartmentInventoryContainerType;      
    // // // typedef  compartmentInventoryContainerType::iterator compartmentInventoryIterator;

	// // // CompartmentInventory();
	// // // ~CompartmentInventory();

	// // // void setPotts3DPtr(Potts3D *_potts);	
	// // // CC3DCellList getClusterCells(long _clusterId);
	// // // void addToInventory(CellG * _cell);
	// // // void removeClusterFromInventory(long _clusterId);
	// // // void removeFromInventory(CellG * _cell);

	// // // compartmentInventoryContainerType & getContainer(){return inventory;}

	// // // //bool reassignClusterId(CellG *,long);
	// // // //iterator part
		// // // std::map<long,CellG *>::size_type getInventorySize(){return inventory.size();}
      // // // int getSize(){return inventory.size();}   
	  // // // compartmentInventoryIterator inventoryBegin(){return inventory.begin();}
      // // // compartmentInventoryIterator inventoryEnd(){return inventory.end();}
      // // // void incrementIterator(compartmentInventoryIterator & _itr){++_itr;}
      // // // void decrementIterator(compartmentInventoryIterator & _itr){--_itr;}


// // // private:
	// // // Potts3D *potts;
	// // // compartmentInventoryContainerType inventory;

// // // };



};

#endif
