#ifndef FIELDEXTRACTORBASE_H
#define FIELDEXTRACTORBASE_H

#include <vector>
#include <map>
#include <string>
#include <Utils/Coordinates3D.h>

#include "FieldExtractorDLLSpecifier.h"


//Notice one can speed up filling up of the Hex lattice data by allocating e.g. hexPOints ot cellType arrays
//instead of inserting values. Inserting causes reallocations and this slows down the task completion

namespace CompuCell3D{

	class FIELDEXTRACTOR_EXPORT  FieldExtractorBase{
	public:

		FieldExtractorBase();
		virtual ~FieldExtractorBase();

		std::vector<int> pointOrder(std::string _plane);
		std::vector<int> dimOrder(std::string _plane);
		Coordinates3D<double> HexCoordXY(unsigned int x,unsigned int y,unsigned int z);


		virtual void fillCellFieldData2D(long _cellTypeArrayAddr , std::string _plane ,  int _pos);
		virtual void fillBorderData2D(long _pointArrayAddr ,long _linesArrayAddr, std::string _plane ,  int _pos){}

		virtual void fillCentroidData2D(long _pointArrayAddr ,long _linesArrayAddr, std::string _plane ,  int _pos){}

		virtual void fillCellFieldData2DHex(long _cellTypeArrayAddr,long _hexCellsArrayAddr ,long _pointsArrayAddr, std::string _plane ,  int _pos){}
		virtual void fillBorderData2DHex(long _pointArrayAddr ,long _linesArrayAddr, std::string _plane ,  int _pos){}

		virtual void fillClusterBorderData2D(long _pointArrayAddr ,long _linesArrayAddr, std::string _plane ,  int _pos){}
		virtual void fillClusterBorderData2DHex(long _pointArrayAddr ,long _linesArrayAddr, std::string _plane ,  int _pos){}

		virtual bool fillConFieldData2D(long _conArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos){return false;}
		virtual bool fillConFieldData2DHex(long _conArrayAddr,long _hexCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){return false;}
        virtual bool fillConFieldData2DCartesian(long _conArrayAddr,long _cartesianCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){return false;}
        
		virtual bool fillScalarFieldData2D(long _conArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos){return false;}
		virtual bool fillScalarFieldData2DHex(long _conArrayAddr,long _hexCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){return false;}
        virtual bool fillScalarFieldData2DCartesian(long _conArrayAddr,long _hexCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){return false;}

		virtual bool fillScalarFieldCellLevelData2D(long _conArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos){return false;}
		virtual bool fillScalarFieldCellLevelData2DHex(long _conArrayAddr,long _hexCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){return false;}
        virtual bool fillScalarFieldCellLevelData2DCartesian(long _conArrayAddr,long _hexCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){return false;}
        
		virtual bool fillVectorFieldData2D(long _pointsArrayIntAddr,long _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos){return false;}
		virtual bool fillVectorFieldData2DHex(long _pointsArrayIntAddr,long _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos){return false;}

		virtual bool fillVectorFieldData3D(long _pointsArrayIntAddr,long _vectorArrayIntAddr,std::string _fieldName){return false;}

		virtual bool fillVectorFieldCellLevelData2D(long _pointsArrayIntAddr,long _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos){return false;}
		virtual bool fillVectorFieldCellLevelData2DHex(long _pointsArrayIntAddr,long _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos){return false;}

		virtual bool fillVectorFieldCellLevelData3D(long _pointsArrayIntAddr,long _vectorArrayIntAddr,std::string _fieldName){return false;}

		virtual bool fillScalarFieldData3D(long _conArrayAddr ,long _cellTypeArrayAddr, std::string _conFieldName,std::vector<int> * _typesInvisibeVec){return false;}

		virtual bool fillScalarFieldCellLevelData3D(long _conArrayAddr ,long _cellTypeArrayAddr, std::string _conFieldName,std::vector<int> * _typesInvisibeVec){return false;}

		virtual std::vector<int> fillCellFieldData3D(long _cellTypeArrayAddr, long _cellIdArrayAddr){return std::vector<int>();}
		virtual bool fillConFieldData3D(long _conArrayAddr ,long _cellTypeArrayAddr, std::string _conFieldName,std::vector<int> * _typesInvisibeVec){return false;}

		void* unmangleSWIGVktPtr(std::string _swigStyleVtkPtr);
		long unmangleSWIGVktPtrAsLong(std::string _swigStyleVtkPtr);


	protected:
		std::vector<Coordinates3D<double> > hexagonVertices;
        std::vector<Coordinates3D<double> > cartesianVertices;
        
        

	};
};

#endif
