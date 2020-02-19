#ifndef FIELDEXTRACTORBASE_H
#define FIELDEXTRACTORBASE_H

#include <vector>
#include <map>
#include <string>
#include <Utils/Coordinates3D.h>
#include "FieldExtractorDLLSpecifier.h"
#include "FieldExtractorTypes.h"

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

		virtual void fillCellFieldData2D(vtk_obj_addr_int_t _cellTypeArrayAddr , std::string _plane ,  int _pos);
		virtual void fillBorderData2D(vtk_obj_addr_int_t _pointArrayAddr , vtk_obj_addr_int_t _linesArrayAddr, std::string _plane ,  int _pos){}

		virtual void fillCentroidData2D(vtk_obj_addr_int_t _pointArrayAddr , vtk_obj_addr_int_t _linesArrayAddr, std::string _plane ,  int _pos){}

		virtual void fillCellFieldData2DHex(vtk_obj_addr_int_t _cellTypeArrayAddr,vtk_obj_addr_int_t _hexCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane ,  int _pos){}
		virtual void fillBorderData2DHex(vtk_obj_addr_int_t _pointArrayAddr ,vtk_obj_addr_int_t _linesArrayAddr, std::string _plane ,  int _pos){}

		virtual void fillClusterBorderData2D(vtk_obj_addr_int_t _pointArrayAddr ,vtk_obj_addr_int_t _linesArrayAddr, std::string _plane ,  int _pos){}
		virtual void fillClusterBorderData2DHex(vtk_obj_addr_int_t _pointArrayAddr ,vtk_obj_addr_int_t _linesArrayAddr, std::string _plane ,  int _pos){}

		virtual bool fillConFieldData2D(vtk_obj_addr_int_t _conArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos){return false;}
		virtual bool fillConFieldData2DHex(vtk_obj_addr_int_t _conArrayAddr,vtk_obj_addr_int_t _hexCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){return false;}
        virtual bool fillConFieldData2DCartesian(vtk_obj_addr_int_t _conArrayAddr,vtk_obj_addr_int_t _cartesianCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){return false;}
        
		virtual bool fillScalarFieldData2D(vtk_obj_addr_int_t _conArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos){return false;}
		virtual bool fillScalarFieldData2DHex(vtk_obj_addr_int_t _conArrayAddr,vtk_obj_addr_int_t _hexCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){return false;}
        virtual bool fillScalarFieldData2DCartesian(vtk_obj_addr_int_t _conArrayAddr,vtk_obj_addr_int_t _cartesianCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){return false;}

		virtual bool fillScalarFieldCellLevelData2D(vtk_obj_addr_int_t _conArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos){return false;}
		virtual bool fillScalarFieldCellLevelData2DHex(vtk_obj_addr_int_t _conArrayAddr,vtk_obj_addr_int_t _hexCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){return false;}
        virtual bool fillScalarFieldCellLevelData2DCartesian(vtk_obj_addr_int_t _conArrayAddr,vtk_obj_addr_int_t _cartesianCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){return false;}
        
		virtual bool fillVectorFieldData2D(vtk_obj_addr_int_t _pointsArrayIntAddr,vtk_obj_addr_int_t _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos){return false;}
		virtual bool fillVectorFieldData2DHex(vtk_obj_addr_int_t _pointsArrayIntAddr,vtk_obj_addr_int_t _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos){return false;}

		virtual bool fillVectorFieldData3D(vtk_obj_addr_int_t _pointsArrayIntAddr,vtk_obj_addr_int_t _vectorArrayIntAddr,std::string _fieldName){return false;}

		virtual bool fillVectorFieldCellLevelData2D(vtk_obj_addr_int_t _pointsArrayIntAddr,vtk_obj_addr_int_t _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos){return false;}
		virtual bool fillVectorFieldCellLevelData2DHex(vtk_obj_addr_int_t _pointsArrayIntAddr,vtk_obj_addr_int_t _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos){return false;}

		virtual bool fillVectorFieldCellLevelData3D(vtk_obj_addr_int_t _pointsArrayIntAddr,vtk_obj_addr_int_t _vectorArrayIntAddr,std::string _fieldName){return false;}

		virtual bool fillScalarFieldData3D(vtk_obj_addr_int_t _conArrayAddr ,vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _conFieldName,std::vector<int> * _typesInvisibeVec){return false;}

		virtual bool fillScalarFieldCellLevelData3D(vtk_obj_addr_int_t _conArrayAddr ,vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _conFieldName,std::vector<int> * _typesInvisibeVec){return false;}

		virtual std::vector<int> fillCellFieldData3D(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _cellIdArrayAddr){return std::vector<int>();}
		virtual bool fillConFieldData3D(vtk_obj_addr_int_t _conArrayAddr ,vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _conFieldName,std::vector<int> * _typesInvisibeVec){return false;}

		void fillNCMaterialDisplayField(vtk_obj_addr_int_t _colorsArrayAddr, vtk_obj_addr_int_t _quantityArrayAddr, vtk_obj_addr_int_t _colors_lutAddr);
		virtual void fillNCMaterialFieldData2D(vtk_obj_addr_int_t _ncmQuantityArrayAddr, std::string _plane, int _pos, int _compSel) {}
		virtual void fillNCMaterialFieldData2DHex(vtk_obj_addr_int_t _ncmQuantityArrayAddr, vtk_obj_addr_int_t _hexCellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane, int _pos, int _compSel) {}
		virtual void fillNCMaterialData2DCartesian(vtk_obj_addr_int_t _ncmQuantityArrayAddr, vtk_obj_addr_int_t _cartesianCellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane, int _pos, int _compSel) {}
		virtual void fillNCMaterialFieldData3D(vtk_obj_addr_int_t _ncmQuantityArrayAddr, int _compSel) {}

	protected:
		std::vector<Coordinates3D<double> > hexagonVertices;
        std::vector<Coordinates3D<double> > cartesianVertices;

	};
};

#endif
