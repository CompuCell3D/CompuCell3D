#ifndef COMPUCELL3DCELLINVENTORYWATCHER_H
#define COMPUCELL3DCELLINVENTORYWATCHER_H

namespace CompuCell3D {

    class CellG;

    /**
    Written by T.J. Sego, Ph.D.
    */

    /**
     * @brief Simple interface for handling cell inventory events. 
     * 
     * An instance of a class that implements this interface can be registered with 
     * a CellInventory instance to receive notifications of when cells are added to and removed 
     * from the CellInventory instance.
     */
    class CellInventoryWatcher {

    public: 

        virtual void onCellAdd(CellG *cell) {}

        virtual void onCellRemove(CellG *cell) {}

    };

};

#endif // COMPUCELL3DCELLINVENTORYWATCHER_H