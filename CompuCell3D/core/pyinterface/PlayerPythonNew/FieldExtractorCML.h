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

namespace CompuCell3D {

    //have to declare here all the classes that will be passed to this class from Python
    //class Potts3D;
    //class Simulator;
    /*class Dim3D;*/

    class FIELDEXTRACTOR_EXPORT FieldExtractorCML : public FieldExtractorBase {
    public:

        FieldExtractorCML();

        ~FieldExtractorCML();


        virtual void fillCellFieldData2D(vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _plane, int _pos);

        virtual void
        fillCellFieldData2DHex(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _hexCellsArrayAddr,
                               vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane, int _pos);

        virtual void
        fillCellFieldData2DCartesian(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _cellsArrayAddr,
                                     vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane, int _pos);

        virtual void
        fillBorder2D(const char *arrayName, vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr,
                     std::string _plane, int _pos);

        virtual void
        fillBorder2DHex(const char *arrayName, vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr,
                        std::string _plane, int _pos);

        virtual void
        fillBorderData2D(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr, std::string _plane,
                         int _pos);

        virtual void
        fillBorderData2DHex(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr, std::string _plane,
                            int _pos);

        virtual void fillClusterBorderData2D(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr,
                                             std::string _plane, int _pos);

        virtual void fillClusterBorderData2DHex(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr,
                                                std::string _plane, int _pos);

        virtual void
        fillCentroidData2D(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr, std::string _plane,
                           int _pos);

        virtual bool
        fillConFieldData2D(vtk_obj_addr_int_t _conArrayAddr, std::string _conFieldName, std::string _plane, int _pos);

        virtual bool fillConFieldData2DHex(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _hexCellsArrayAddr,
                                           vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName,
                                           std::string _plane, int _pos);

        virtual bool
        fillConFieldData2DCartesian(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cartesianCellsArrayAddr,
                                    vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName, std::string _plane,
                                    int _pos);

        virtual bool
        fillScalarFieldData2D(vtk_obj_addr_int_t _conArrayAddr, std::string _conFieldName, std::string _plane,
                              int _pos);

        virtual bool fillScalarFieldData2DHex(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _hexCellsArrayAddr,
                                              vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName,
                                              std::string _plane, int _pos);

        virtual bool
        fillScalarFieldData2DCartesian(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cartesianCellsArrayAddr,
                                       vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName,
                                       std::string _plane, int _pos);

        virtual bool
        fillScalarFieldCellLevelData2D(vtk_obj_addr_int_t _conArrayAddr, std::string _conFieldName, std::string _plane,
                                       int _pos);

        virtual bool
        fillScalarFieldCellLevelData2DHex(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _hexCellsArrayAddr,
                                          vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName,
                                          std::string _plane, int _pos);

        virtual bool fillScalarFieldCellLevelData2DCartesian(vtk_obj_addr_int_t _conArrayAddr,
                                                             vtk_obj_addr_int_t _cartesianCellsArrayAddr,
                                                             vtk_obj_addr_int_t _pointsArrayAddr,
                                                             std::string _conFieldName, std::string _plane, int _pos);

        virtual bool
        fillVectorFieldData2D(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr,
                              std::string _fieldName, std::string _plane, int _pos);

        virtual bool
        fillVectorFieldData2DHex(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr,
                                 std::string _fieldName, std::string _plane, int _pos);

        virtual bool
        fillVectorFieldData3D(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr,
                              std::string _fieldName);

        virtual bool
        fillVectorFieldData3DHex(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr,
                                 std::string _fieldName);

        virtual bool
        fillVectorFieldCellLevelData2D(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr,
                                       std::string _fieldName, std::string _plane, int _pos);

        virtual bool fillVectorFieldCellLevelData2DHex(vtk_obj_addr_int_t _pointsArrayIntAddr,
                                                       vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName,
                                                       std::string _plane, int _pos);


        virtual bool
        fillVectorFieldCellLevelData3D(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr,
                                       std::string _fieldName);

        virtual bool fillVectorFieldCellLevelData3DHex(vtk_obj_addr_int_t _pointsArrayIntAddr,
                                                       vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName);

        virtual bool fillScalarFieldData3D(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cellTypeArrayAddr,
                                           std::string _conFieldName, std::vector<int> *_typesInvisibeVec,
                                           bool type_indicator_only);

        virtual bool
        fillScalarFieldCellLevelData3D(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cellTypeArrayAddr,
                                       std::string _conFieldName, std::vector<int> *_typesInvisibeVec,
                                       bool type_indicator_only);

        virtual std::vector<int>
        fillCellFieldData3D(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _cellIdArrayAddr,
                            bool extractOuterShellOnly = false);

        virtual bool fillConFieldData3D(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cellTypeArrayAddr,
                                        std::string _conFieldName, std::vector<int> *_typesInvisibeVec,
                                        bool type_indicator_only
                                                );



        virtual void fillCellFieldGlyphs2D(
                vtk_obj_addr_int_t centroids_array_addr,
                vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                vtk_obj_addr_int_t cell_type_array_addr,
                std::string plane, int pos);

        virtual void fillConFieldGlyphs2D(
                std::string con_field_name,
                vtk_obj_addr_int_t centroids_array_addr,
                vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                vtk_obj_addr_int_t scalar_value_at_com_addr,
                std::string plane, int pos);

        virtual void fillScalarFieldGlyphs2D(
                std::string con_field_name,
                vtk_obj_addr_int_t centroids_array_addr,
                vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                vtk_obj_addr_int_t scalar_value_at_com_addr,
                std::string plane, int pos);

        virtual void fillScalarFieldCellLevelGlyphs2D(
                std::string con_field_name,
                vtk_obj_addr_int_t centroids_array_addr,
                vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                vtk_obj_addr_int_t scalar_value_at_com_addr,
                std::string plane, int pos);

        virtual std::vector<int> fillCellFieldGlyphs3D(vtk_obj_addr_int_t centroids_array_addr,
                                                       vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                                                       vtk_obj_addr_int_t cell_type_array_addr,
                                                       std::vector<int> *types_invisibe_vec,
                                                       bool extractOuterShellOnly = false);

        std::vector<int> fillConFieldGlyphs3D(std::string con_field_name,
                                                              vtk_obj_addr_int_t centroids_array_addr,
                                                              vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                                                              vtk_obj_addr_int_t scalar_value_at_com_addr,
                                                              std::vector<int> *types_invisibe_vec,
                                                              bool extractOuterShellOnly=false);

        std::vector<int> fillScalarFieldGlyphs3D(std::string con_field_name,
                                              vtk_obj_addr_int_t centroids_array_addr,
                                              vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                                              vtk_obj_addr_int_t scalar_value_at_com_addr,
                                              std::vector<int> *types_invisibe_vec,
                                              bool extractOuterShellOnly=false);

        std::vector<int> fillScalarFieldCellLevelGlyphs3D(std::string con_field_name,
                                                 vtk_obj_addr_int_t centroids_array_addr,
                                                 vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                                                 vtk_obj_addr_int_t scalar_value_at_com_addr,
                                                 std::vector<int> *types_invisibe_vec,
                                                 bool extractOuterShellOnly=false);

        virtual bool fillLinksField2D(vtk_obj_addr_int_t points_array_addr, 
                                      vtk_obj_addr_int_t lines_array_addr, 
                                      const std::string &plane,
                                      const int &pos,
                                      const int &margin=1) override;

        virtual bool fillLinksField3D(vtk_obj_addr_int_t points_array_addr, 
                                      vtk_obj_addr_int_t lines_array_addr) override;

        virtual bool readVtkStructuredPointsData(vtk_obj_addr_int_t _structuredPointsReaderAddr);

        void setFieldDim(Dim3D _dim);

        Dim3D getFieldDim();

        void setSimulationData(vtk_obj_addr_int_t _structuredPointsAddr);

        long pointIndex(short _x, short _y, short _z);

        long indexPoint3D(Point3D pt);

    private:
        Dim3D fieldDim;
        int zDimFactor, yDimFactor;
        vtkStructuredPoints *lds;
        typedef int (FieldExtractorCML::*type_fcn_ptr_t)(int type);
        FieldExtractorCML::type_fcn_ptr_t type_fcn_ptr;

    };

};


#endif
