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


  class CACell;

class CASHARED_EXPORT CAManager{
    public:
        CAManager();
        ~CAManager();

        void createCellField(const Dim3D & _dim);
		void destroyCell(CACell *  _cell, bool _flag=true);
		CACell * createAndPositionCell(const Point3D &pt, long _clusterId=-1);
        CACell * createCell(long _clusterId=-1);
		void registerCellFieldChangeWatcher(CACellFieldChangeWatcher * _watcher);
		//WatchableField3D<CACell *> * getCellField();
        Field3D<CACell *> * getCellField();
		void positionCell(const Point3D &_pt,CACell *  _cell);
		CACellInventory * getCellInventory();

		//old simulator interface namning convention
		int getNumSteps();
		void setNumSteps(int _numSteps);
		int getCurrentStep();
		void step(int i);
		std::vector<std::string> getConcentrationFieldNameVector();

		void cleanAfterSimulation();


		void setCellToDelete(CACell * _cell); //sets ptr of a cell to be deleted

		

		void cleanup(); //used to delete cells

    protected:
        WatchableField3D<CACell *> *cellField;
		CACellInventory cellInventory;
		long recentlyCreatedCellId;
		long recentlyCreatedClusterId;
		int numSteps;
		int currentStep;
		CACell * cellToDelete;

		
};
};
#endif