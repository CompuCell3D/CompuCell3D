#ifndef PYCOMPUCELLOBJADAPTER_H
#define PYCOMPUCELLOBJADAPTER_H

#include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/Field3D/Point3D.h>
#include <PublicUtilities/ParallelUtilsOpenMP.h>
#include <vector>
#include <Python.h>


namespace CompuCell3D{
   class Simulator;
   class Potts3D;
   

   class PyCompuCellObjAdapter{
   protected:
		ParallelUtilsOpenMP * pUtils;
   
   public:
	  PyCompuCellObjAdapter();
	  virtual ~PyCompuCellObjAdapter(){};
	  
     void setPotts(Potts3D * _potts);
     void setSimulator(Simulator * _sim);
     void registerPyObject(PyObject * _pyObject);

     bool isNewCellValid();
     bool isOldCellValid();
     bool isCellMedium(CellG * cell);
	 CellG * getNewCell();
	 CellG * getOldCell();
	 Point3D  getFlipNeighbor();
	 Point3D  getChangePoint();
	 CellG::CellType_t getNewType();


     //bool isNewCellValid(){return newCell;}
     //bool isOldCellValid(){return oldCell;}
     //bool isCellMedium(CellG * cell){return !cell;}




     Simulator *sim;
     Potts3D *potts;
	 



    protected:
     std::vector<PyObject *> vecPyObject;

	 Point3D changePoint;
     Point3D flipNeighbor;
     CellG *newCell;
     CellG *oldCell;


	 //vectorized variables for OpenMP
	 std::vector<CellG *> newCellVec;
	 std::vector<CellG *> oldCellVec;
	 std::vector<Point3D> flipNeighborVec;
	 std::vector<Point3D> changePointVec;
	 std::vector<CellG::CellType_t> newTypeVec;
   };


};
#endif
