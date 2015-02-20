#ifndef FIELDEXTRACTORCML_H
#define FIELDEXTRACTORCML_H

#include <vector>
#include <map>
#include <string>
#include <Utils/Coordinates3D.h>
//#include "FieldStorage.h"
#include <CompuCell3D/Field3D/Point3D.h>
#include <CompuCell3D/Field3D/Dim3D.h>

//#include <CompuCell3D/Potts3D/Cell.h>

#include "FieldExtractorBase.h"
#include "FieldExtractorDLLSpecifier.h"

class FieldStorage;
class vtkIntArray;
class vtkDoubleArray;
class vtkFloatArray;
class vtkPoints;
class vtkCellArray;
class vtkStructuredPoints;
class vtkStructuredPointsReader;
class vtkObject;



//Notice one can speed up filling up of the Hex lattice data by allocating e.g. hexPOints ot cellType arrays
//instead of inserting values. Inserting causes reallocations and this slows down the task completion

namespace CompuCell3D{

	//have to declare here all the classes that will be passed to this class from Python
	//class Potts3D;
	//class Simulator;
	/*class Dim3D;*/

	class FIELDEXTRACTOR_EXPORT FieldExtractorCML:public FieldExtractorBase{
	public:

		FieldExtractorCML();
		~FieldExtractorCML();


		virtual void fillCellFieldData2D(long _cellTypeArrayAddr , std::string _plane ,  int _pos);
	    virtual void fillCellFieldData2DHex(long _cellTypeArrayAddr,long _hexCellsArrayAddr ,long _pointsArrayAddr, std::string _plane ,  int _pos);

	    virtual void fillBorder2D(const char* arrayName, long _pointArrayAddr ,long _linesArrayAddr, std::string _plane ,  int _pos);
	    virtual void fillBorder2DHex(const char* arrayName, long _pointArrayAddr ,long _linesArrayAddr, std::string _plane ,  int _pos);

		virtual void fillBorderData2D(long _pointArrayAddr ,long _linesArrayAddr, std::string _plane ,  int _pos);
		virtual void fillBorderData2DHex(long _pointArrayAddr ,long _linesArrayAddr, std::string _plane ,  int _pos);

		virtual void fillClusterBorderData2D(long _pointArrayAddr ,long _linesArrayAddr, std::string _plane ,  int _pos);
		virtual void fillClusterBorderData2DHex(long _pointArrayAddr ,long _linesArrayAddr, std::string _plane ,  int _pos);

		virtual void fillCentroidData2D(long _pointArrayAddr ,long _linesArrayAddr, std::string _plane ,  int _pos);

		virtual bool fillConFieldData2D(long _conArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos);
		virtual bool fillConFieldData2DHex(long _conArrayAddr,long _hexCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos);
        virtual bool fillConFieldData2DCartesian(long _conArrayAddr,long _cartesianCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos);

	    virtual bool fillScalarFieldData2D(long _conArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos);
	    virtual bool fillScalarFieldData2DHex(long _conArrayAddr,long _hexCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos);
        virtual bool fillScalarFieldData2DCartesian(long _conArrayAddr,long _cartesianCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos);        

		virtual bool fillScalarFieldCellLevelData2D(long _conArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos);
		virtual bool fillScalarFieldCellLevelData2DHex(long _conArrayAddr,long _hexCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos);
        virtual bool fillScalarFieldCellLevelData2DCartesian(long _conArrayAddr,long _cartesianCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos);

		virtual bool fillVectorFieldData2D(long _pointsArrayIntAddr,long _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos);
		virtual bool fillVectorFieldData2DHex(long _pointsArrayIntAddr,long _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos);

		virtual bool fillVectorFieldData3D(long _pointsArrayIntAddr,long _vectorArrayIntAddr,std::string _fieldName);

		virtual bool fillVectorFieldCellLevelData2D(long _pointsArrayIntAddr,long _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos);
		virtual bool fillVectorFieldCellLevelData2DHex(long _pointsArrayIntAddr,long _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos);

		virtual bool fillVectorFieldCellLevelData3D(long _pointsArrayIntAddr,long _vectorArrayIntAddr,std::string _fieldName);

	    virtual bool fillScalarFieldData3D(long _conArrayAddr ,long _cellTypeArrayAddr, std::string _conFieldName,std::vector<int> * _typesInvisibeVec);

		virtual bool fillScalarFieldCellLevelData3D(long _conArrayAddr ,long _cellTypeArrayAddr, std::string _conFieldName,std::vector<int> * _typesInvisibeVec);

		virtual std::vector<int> fillCellFieldData3D(long _cellTypeArrayAddr, long _cellIdArrayAddr);
		virtual bool fillConFieldData3D(long _conArrayAddr ,long _cellTypeArrayAddr, std::string _conFieldName,std::vector<int> * _typesInvisibeVec);

        virtual bool readVtkStructuredPointsData(long _structuredPointsReaderAddr);
        
		void setFieldDim(Dim3D _dim);
		Dim3D getFieldDim();
		void setSimulationData(long _structuredPointsAddr);
		long pointIndex(short _x,short _y,short _z);
		long indexPoint3D(Point3D pt);
	private:
		Dim3D fieldDim;
		int zDimFactor,yDimFactor;
		vtkStructuredPoints * lds;
	};

};



#endif
