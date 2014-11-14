#ifndef CAFIELDEXTRACTOR_H
#define CAFIELDEXTRACTOR_H

#include <vector>
#include <map>
#include <string>
#include <Utils/Coordinates3D.h>
// #include "FieldStorage.h"

// #include <CompuCell3D/Potts3D/Cell.h>

// #include "FieldExtractorBase.h"

#include "CAFieldUtilsDLLSpecifier.h"

class FieldStorage;
class vtkIntArray;
class vtkDoubleArray;
class vtkFloatArray;
class vtkPoints;
class vtkCellArray;
class vtkObject;


//Notice one can speed up filling up of the Hex lattice data by allocating e.g. hexPOints ot cellType arrays
//instead of inserting values. Inserting causes reallocations and this slows down the task completion

namespace CompuCell3D{

	//have to declare here all the classes that will be passed to this class from Python
	class CAManager;
	
	class Dim3D;

	class CAFIELDUTILS_EXPORT CAFieldExtractor{
	public:
		CAManager * caManager;		
		CAFieldExtractor();
		virtual ~CAFieldExtractor();

		// void setFieldStorage(FieldStorage * _fsPtr){fsPtr=_fsPtr;}
		// FieldStorage * getFieldStorage(FieldStorage * _fsPtr){return fsPtr;}

		void extractCellField();

		virtual void fillCellFieldData2D(long _cellTypeArrayAddr , long _centroidPointsAddr, long _scaleRadiusArrayAddr, std::string _plane ,  int _pos);
        virtual void fillCellFieldData3D(long _cellTypeArrayAddr , long _centroidPointsAddr, long _scaleRadiusArrayAddr);

		virtual bool fillScalarFieldData2DCartesian(long _conArrayAddr,long _cartesianCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos);

		// virtual void fillCellFieldData2DHex_old(long _cellTypeArrayAddr ,long _pointsArrayAddr, std::string _plane ,  int _pos);
        
	    // virtual void fillCellFieldData2DHex(long _cellTypeArrayAddr,long _hexCellsArrayAddr ,long _pointsArrayAddr, std::string _plane ,  int _pos);

		// virtual void fillBorderData2D(long _pointArrayAddr ,long _linesArrayAddr, std::string _plane ,  int _pos);
		// virtual void fillBorderData2DHex(long _pointArrayAddr ,long _linesArrayAddr, std::string _plane ,  int _pos);

		// virtual void fillClusterBorderData2D(long _pointArrayAddr ,long _linesArrayAddr, std::string _plane ,  int _pos);
		// virtual void fillClusterBorderData2DHex(long _pointArrayAddr ,long _linesArrayAddr, std::string _plane ,  int _pos);

		// virtual void fillCentroidData2D(long _pointArrayAddr ,long _linesArrayAddr, std::string _plane ,  int _pos);

		// virtual bool fillConFieldData2D(long _conArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos);
		// virtual bool fillConFieldData2DHex(long _conArrayAddr,long _hexCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos);
        // // virtual bool fillConFieldData2DCartesian(long _conArrayAddr,long _cartesianCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos);
        // // {return false;}
        // virtual bool fillConFieldData2DCartesian(long _conArrayAddr,long _cartesianCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos);

	    // virtual bool fillScalarFieldData2D(long _conArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos);
	    // virtual bool fillScalarFieldData2DHex(long _conArrayAddr,long _hexCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos);
        // virtual bool fillScalarFieldData2DCartesian(long _conArrayAddr,long _hexCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos);

		// virtual bool fillScalarFieldCellLevelData2D(long _conArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos);
		// virtual bool fillScalarFieldCellLevelData2DHex(long _conArrayAddr,long _hexCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos);
        // virtual bool fillScalarFieldCellLevelData2DCartesian(long _conArrayAddr,long _hexCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos);
		// virtual bool fillScalarFieldCellLevelData3D(long _conArrayAddr ,long _cellTypeArrayAddr, std::string _conFieldName,std::vector<int> * _typesInvisibeVec);

		// virtual bool fillVectorFieldData2D(long _pointsArrayIntAddr,long _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos);
		// virtual bool fillVectorFieldData2DHex(long _pointsArrayIntAddr,long _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos);
		// virtual bool fillVectorFieldData3D(long _pointsArrayIntAddr,long _vectorArrayIntAddr,std::string _fieldName);

		// virtual bool fillVectorFieldCellLevelData2D(long _pointsArrayIntAddr,long _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos);
		// virtual bool fillVectorFieldCellLevelData2DHex(long _pointsArrayIntAddr,long _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos);
		// virtual bool fillVectorFieldCellLevelData3D(long _pointsArrayIntAddr,long _vectorArrayIntAddr,std::string _fieldName);

	    // virtual bool fillScalarFieldData3D(long _conArrayAddr ,long _cellTypeArrayAddr, std::string _conFieldName,std::vector<int> * _typesInvisibeVec);

		// virtual std::vector<int> fillCellFieldData3D(long _cellTypeArrayAddr, long _cellIdArrayAddr);
		// virtual bool fillConFieldData3D(long _conArrayAddr ,long _cellTypeArrayAddr, std::string _conFieldName,std::vector<int> * _typesInvisibeVec);

		// void setVtkObj(void * _vtkObj);
		// void setVtkObjInt(long _vtkObjAddr);
		// vtkIntArray * produceVtkIntArray();

		// int * produceArray(int _size);
        
        void* unmangleSWIGVktPtr(std::string _swigStyleVtkPtr);
        long unmangleSWIGVktPtrAsLong(std::string _swigStyleVtkPtr);
		std::vector<int> pointOrder(std::string _plane);
		std::vector<int> dimOrder(std::string _plane);

		void init(CAManager * _caManager);
	private:
		std::vector<Coordinates3D<double> > cartesianVertices;
		// FieldStorage * fsPtr;
	};
};

#endif
