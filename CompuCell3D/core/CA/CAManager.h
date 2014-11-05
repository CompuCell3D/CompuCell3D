#ifndef CAMANAGER_H
#define CAMANAGER_H

 #include <BasicUtils/BasicDynamicClassFactory.h>

 #include <BasicUtils/BasicClassAccessor.h>
 #include <BasicUtils/BasicClassGroupFactory.h>
 #include <BasicUtils/BasicClassGroup.h>
#include "CADLLSpecifier.h"
#include "CACellInventory.h"
#include <vector>
#include <set>
#include <string>

#include <Python.h>


class BasicRandomNumberGeneratorNonStatic ;


namespace CompuCell3D {
class Dim3D;
class Point3D;
class CACellStackFieldChangeWatcher;

  template<typename T>
  class WatchableField3D;

  template<typename T>
  class Field3DImpl;

  template<typename T>
  class Field3D;

  class BoundaryStrategy;

  
  class ProbabilityFunction ;
  class CACell;
  class CACellStack;

class CASHARED_EXPORT CAManager{

    public:
        CAManager();
        ~CAManager();

        void createCellField(const Dim3D & _dim);
		void destroyCell(CACell *  _cell, bool _flag=true);
		//CACell * createAndPositionCell(const Point3D &pt, long _clusterId=-1);
		CACell * createAndPositionCellS(const Point3D & pt, long _clusterId=-1);

		//cell creation functions
		void registerClassAccessor(BasicClassAccessorBase *_accessor);
		void registerPythonAttributeAdderObject(PyObject *_attrAdder);
        CACell * createCell(long _clusterId=-1);
		void registerCACellStackFieldChangeWatcher(CACellStackFieldChangeWatcher * _watcher);
		Field3D<CACellStack *> * getCellField();
        
		Field3D<CACellStack *> * getCellFieldS();

		//void positionCell(const Point3D &_pt,CACell *  _cell);
		void positionCellS(const Point3D &_pt,CACell *  _cell); //positions cells in the stack field




		CACellInventory * getCellInventory();

		void setCellCarryingCapacity(int _depth);
		int getCellCarryingCapacity();

		//old simulator interface namning convention
		int getNumSteps();
		void setNumSteps(int _numSteps);
		int getCurrentStep();
		void step(int i);
		

		void cleanAfterSimulation();


		void setCellToDelete(CACell * _cell); //sets ptr of a cell to be deleted

		BoundaryStrategy * getBoundaryStrategy();

		void cleanup(); //used to delete cells


		//Simulation utils
		void registerProbabilityFunction(ProbabilityFunction *_fcn);
		void setNeighborOrder(int _no);
		int getNeighborOrder();

		void runCAAlgorithm(int _mcs);

		void setMaxProb(float _maxProb);
		float getMaxProb();

		//concentration fields
		void registerConcentrationField(std::string _name,Field3D<float>* _fieldPtr);
		std::map<std::string,Field3D<float>*> & getConcentrationFieldNameMap();
		Field3D<float>* getConcentrationField(std::string _fieldName);
		std::vector<std::string> getConcentrationFieldNameVector();

		void setFrozenType(unsigned char _type);
		std::vector<unsigned char> getFrozenTypeVec();
		bool isFrozen(unsigned char _type);

	    std::string getTypeName(const char _type) const;
		unsigned char getTypeId(const std::string _typeName) const;
		void setTypeNameTypeId(std::string _typeName, unsigned char _typeId);
        void clearCellTypeInfo();





    protected:
		BasicClassGroupFactory cellFactoryGroup;

        //WatchableField3D<CACell *> *cellField;
		WatchableField3D<CACellStack *> *cellFieldS;
		int cellCarryingCapacity;
		float maxProb;

		CACellInventory cellInventory;
		long recentlyCreatedCellId;
		long recentlyCreatedClusterId;
		int numSteps;
		int currentStep;
		CACell * cellToDelete;
		BoundaryStrategy *boundaryStrategy;
		int neighborOrder;

		std::vector<ProbabilityFunction *> probFcnRegistry;

		BasicRandomNumberGeneratorNonStatic * rand; 

		std::map<std::string,Field3D<float>*> concentrationFieldNameMap;
		std::vector<PyObject *> attrAdderPyObjectVec;

		std::vector<CACellStackFieldChangeWatcher *> caCellStackWatcherRegistry;
		std::set<unsigned char> frozenTypesSet;
		std::map<std::string, unsigned char> typeName2Id;
		std::map<unsigned char, std::string> id2TypeName;

		
};
};
#endif