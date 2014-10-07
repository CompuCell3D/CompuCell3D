#ifndef CAMANAGER_H
#define CAMANAGER_H


#include "CADLLSpecifier.h"
#include "CACellInventory.h"


namespace CompuCell3D {
class Dim3D;
class Point3D;

  template<typename T>
  class WatchableField3D;

  template<typename T>
  class Field3DImpl;
  
  class CACell;

class CASHARED_EXPORT CAManager{
    public:
        CAManager();
        ~CAManager();

        void createCellField(const Dim3D & _dim);
		void destroyCell(CACell *  _cell, bool _flag);
		CACell * createAndPositionCell(const Point3D &pt, long _clusterId=-1);
        CACell * createCell(long _clusterId=-1);
        
    protected:
        WatchableField3D<CACell *> *cellField;
		CACellInventory cellInventory;
		long recentlyCreatedCellId;
		long recentlyCreatedClusterId;
};
};
#endif