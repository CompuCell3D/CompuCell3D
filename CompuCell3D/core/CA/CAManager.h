#ifndef CAMANAGER_H
#define CAMANAGER_H


#include "CADLLSpecifier.h"
#include "CACellInventory.h"
#include <vector>
#include <string>


namespace CompuCell3D {
class Dim3D;
class Point3D;
class CACellFieldChangeWatcher;

  template<typename T>
  class WatchableField3D;

  template<typename T>
  class Field3DImpl;

  template<typename T>
  class Field3D;

  class BoundaryStrategy;


  class CACell;
  class CACellStack;

class CASHARED_EXPORT CAManager{

    public:
        CAManager();
        ~CAManager();

        void createCellField(const Dim3D & _dim);
		void destroyCell(CACell *  _cell, bool _flag=true);
		CACell * createAndPositionCell(const Point3D &pt, long _clusterId=-1);
		CACell * createAndPositionCellS(const Point3D & pt, long _clusterId=-1);

        CACell * createCell(long _clusterId=-1);
		void registerCellFieldChangeWatcher(CACellFieldChangeWatcher * _watcher);
		//WatchableField3D<CACell *> * getCellField();
        Field3D<CACell *> * getCellField();
		Field3D<CACellStack *> * getCellFieldS();

		void positionCell(const Point3D &_pt,CACell *  _cell);
		void positionCellS(const Point3D &_pt,CACell *  _cell); //positions cells in the stack field
		CACellInventory * getCellInventory();

		void setCellCarryingCapacity(int _depth);
		int getCellCarryingCapacity();

		//old simulator interface namning convention
		int getNumSteps();
		void setNumSteps(int _numSteps);
		int getCurrentStep();
		void step(int i);
		std::vector<std::string> getConcentrationFieldNameVector();

		void cleanAfterSimulation();


		void setCellToDelete(CACell * _cell); //sets ptr of a cell to be deleted

		BoundaryStrategy * getBoundaryStrategy();

		void cleanup(); //used to delete cells

    protected:
        WatchableField3D<CACell *> *cellField;
		WatchableField3D<CACellStack *> *cellFieldS;
		int cellCarryingCapacity;

		CACellInventory cellInventory;
		long recentlyCreatedCellId;
		long recentlyCreatedClusterId;
		int numSteps;
		int currentStep;
		CACell * cellToDelete;
		BoundaryStrategy *boundaryStrategy;

		
};
};
#endif