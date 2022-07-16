#ifndef COMPUCELL3DCELLINVENTORY_H
#define COMPUCELL3DCELLINVENTORY_H

#include <set>
#include <vector>
#include <map>
//NOTE: compartment inventory should be changed to cluster inventory to avoid name confusion
#include "CellInventoryWatcher.h"


namespace CompuCell3D {

    class CellG;

    class Potts3D;

/**
@author m
*/

    class CC3DCellList : public std::vector<CellG *> {
//may add interface later if necessary
    public:
        typedef std::vector<CellG *>::iterator CC3DCellListIterator_t;

        virtual ~CC3DCellList();

        //added it to make python interfacing a bit easier - should implement separate interface in C++ anyway
        virtual std::vector<CellG *> *getBaseClass() {
            return (std::vector < CellG * > *)
            this;
        }


    };
//this inventory will have to be rewritten using less ad-hoc coding styles....

    class CompartmentInventory {

    public:
        typedef std::map<long, CellG *> compartmentListContainerType;
        typedef std::map<long, CellG *>::iterator compartmentListIterator;
        typedef std::map<long, compartmentListContainerType> compartmentInventoryContainerType;
        typedef compartmentInventoryContainerType::iterator compartmentInventoryIterator;

        CompartmentInventory();

        ~CompartmentInventory();

        void setPotts3DPtr(Potts3D *_potts);

        CC3DCellList getClusterCells(long _clusterId);

        void addToInventory(CellG *_cell);

        void removeClusterFromInventory(long _clusterId);

        void removeFromInventory(CellG *_cell);

        compartmentInventoryContainerType &getContainer() { return inventory; }

        //iterator part
        std::map<long, CellG *>::size_type getInventorySize() { return inventory.size(); }

        int getSize() { return inventory.size(); }

        compartmentInventoryIterator inventoryBegin() { return inventory.begin(); }

        compartmentInventoryIterator inventoryEnd() { return inventory.end(); }

        void incrementIterator(compartmentInventoryIterator &_itr) { ++_itr; }

        void decrementIterator(compartmentInventoryIterator &_itr) { --_itr; }


    private:
        Potts3D *potts;
        compartmentInventoryContainerType inventory;

    };

    class CellIdentifier {
    public:
        CellIdentifier(long _cellId = 0, long _clusterId = 0) : cellId(_cellId), clusterId(_clusterId) {}

        long cellId;
        long clusterId;

        ///have to define < operator if using a class in the set and no < operator is defined for this class
        bool operator<(const CellIdentifier &_rhs) const {
            //return clusterId < _rhs.clusterId || (!(_rhs.clusterId < clusterId ) && cellId < _rhs.cellId);
            // this ordering (first cell id, then cluster id) is necessary to get attemptFetchingCellById function working properly
            //return cellId < _rhs.cellId || (!(cellId < _rhs.cellId) && clusterId < _rhs.clusterId );//old and wrong implementation of comparison operator might give side effects on windows - it can crash CC3D or in some cases windows OS entirely
            return cellId < _rhs.cellId || (!(_rhs.cellId < cellId) && clusterId < _rhs.clusterId);

        }

    };

    class CellInventory {
    public:
        typedef std::map<CellIdentifier, CellG *> cellInventoryContainerType;
        //typedef  std::set<CellG *> cellInventoryContainerType;
        typedef cellInventoryContainerType::iterator cellInventoryIterator;
        typedef std::map<long, CellG *> cellListByType_t;


        CellInventory();

        virtual ~CellInventory();

        virtual void addToInventory(CellG *_cell);

        virtual void removeFromInventory(CellG *_cell);

        std::map<long, CellG *>::size_type getCellInventorySize() { return inventory.size(); }

        int getSize() { return inventory.size(); }

        cellInventoryIterator cellInventoryBegin() { return inventory.begin(); }

        cellInventoryIterator cellInventoryEnd() { return inventory.end(); }

        void incrementIterator(cellInventoryIterator &_itr) { ++_itr; }

        void decrementIterator(cellInventoryIterator &_itr) { --_itr; }

        bool reassignClusterId(CellG *, long);

        cellInventoryIterator find(CellG *_cell);

        cellInventoryIterator find(long _id) { return inventory.find(_id); };

        cellInventoryContainerType &getContainer() { return inventory; }

        void setPotts3DPtr(Potts3D *_potts);

        CellG *getCellById(long _id); //obsolete
        CellG *getCellByIds(long _id, long clusterId);

        CellG *attemptFetchingCellById(long _id);


        CellG *getCell(cellInventoryIterator &_itr) { return _itr->second; }

        CC3DCellList getClusterCells(long _clusterId);

        void initCellInventoryByType(cellListByType_t *_inventoryByTypePtr,
                                     unsigned char _type); //the return variable is the same as the second argument
        void initCellInventoryByMultiType(cellListByType_t *_inventoryByTypePtr, std::vector<int> *_typeVecPtr);

        CompartmentInventory &getClusterInventory() { return compartmentInventory; }

        void cleanInventory();

        void registerWatcher(CellInventoryWatcher *watcher);

        void unregisterWatcher(CellInventoryWatcher *watcher);

    private:
        cellInventoryContainerType inventory;
        Potts3D *potts;
        CompartmentInventory compartmentInventory;
        std::vector<CellInventoryWatcher*> watchers;

    };

};

#endif
