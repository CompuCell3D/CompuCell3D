#ifndef FIELDEXTRACTOR_H
#define FIELDEXTRACTOR_H

#include <vector>
#include <map>
#include <string>
#include <Utils/Coordinates3D.h>
#include "FieldStorage.h"


#include <CompuCell3D/Potts3D/Cell.h>

#include "FieldExtractorBase.h"

#include "FieldExtractorDLLSpecifier.h"

class FieldStorage;

class vtkIntArray;

class vtkDoubleArray;

class vtkFloatArray;

class vtkPoints;

class vtkCellArray;

class vtkObject;




//Notice one can speed up filling up of the Hex lattice data by allocating e.g. hexPOints ot cellType arrays
//instead of inserting values. Inserting causes reallocations and this slows down the task completion

namespace CompuCell3D {

    //have to declare here all the classes that will be passed to this class from Python
    class Potts3D;

    class Simulator;

    class Dim3D;

    class NeighborTracker;

    class ParallelUtilsOpenMP;

    template <typename T>
    class Field3D;


    class FIELDEXTRACTOR_EXPORT FieldExtractor : public FieldExtractorBase {
    public :
        Potts3D *potts;
        Simulator *sim;

        FieldExtractor();

        ~FieldExtractor();

        void setFieldStorage(FieldStorage *_fsPtr) { fsPtr = _fsPtr; }

        FieldStorage *getFieldStorage(FieldStorage *_fsPtr) { return fsPtr; }

        void extractCellField();

        virtual void fillCellFieldData2D(vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _plane, int _pos);

        virtual void
        fillCellFieldData2DCartesian(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _cellsArrayAddr,
                                     vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane, int _pos);

        virtual void
        fillCellFieldData2DHex(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _hexCellsArrayAddr,
                               vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane, int _pos);

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

        // virtual bool fillConFieldData2DCartesian(vtk_obj_addr_int_t _conArrayAddr,vtk_obj_addr_int_t _cartesianCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos);
        // {return false;}
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

        virtual bool fillConFieldData2DCartesianFlex(vtk_obj_addr_int_t _conArrayAddr,
                                                     vtk_obj_addr_int_t _cartesianCellsArrayAddr,
                                                     vtk_obj_addr_int_t _pointsArrayAddr,
                                                     std::string _conFieldName,
                                                     std::string _plane, int _pos) ;

        template <typename T>
        bool fillConFieldData2DCartesianTyped(vtkDoubleArray *conArray, vtkCellArray *_cartesianCellsArray,
                                                              vtkPoints *_pointsArray, Field3D<T> *conFieldPtr,
                                                              std::string _plane, int _pos);
//        {
//            if (!conFieldPtr) return false;
//
//            Field3D<CellG *> *cellFieldG = potts->getCellFieldG();
//            Dim3D fieldDim = cellFieldG->getDim();
//
//            std::vector<int> fieldDimVec = { fieldDim.x, fieldDim.y, fieldDim.z };
//            std::vector<int> pointOrderVec = pointOrder(_plane);
//            std::vector<int> dimOrderVec = dimOrder(_plane);
//
//            std::vector<int> dim(3, 0);
//            dim[0] = fieldDimVec[dimOrderVec[0]];
//            dim[1] = fieldDimVec[dimOrderVec[1]];
//            dim[2] = fieldDimVec[dimOrderVec[2]];
//
//            int offset = 0;
//            Point3D pt;
//            std::vector<int> ptVec(3, 0);
//            T con;
//            long pc = 0;
//
//            for (int j = 0; j < dim[1]; ++j) {
//                for (int i = 0; i < dim[0]; ++i) {
//                    ptVec[0] = i;
//                    ptVec[1] = j;
//                    ptVec[2] = _pos;
//
//                    pt.x = ptVec[pointOrderVec[0]];
//                    pt.y = ptVec[pointOrderVec[1]];
//                    pt.z = ptVec[pointOrderVec[2]];
//
//                    if (i == dim[0] || j == dim[1]) {
//                        con = 0;
//                    } else {
//                        con = conFieldPtr->get(pt);
//                    }
//
//                    Coordinates3D<double> coords(ptVec[0], ptVec[1], 0);
//
//                    for (int idx = 0; idx < 4; ++idx) {
//                        Coordinates3D<double> cartesianVertex = cartesianVertices[idx] + coords;
//                        _pointsArray->InsertNextPoint(cartesianVertex.x, cartesianVertex.y, 0.0);
//                    }
//
//                    pc += 4;
//                    vtkIdType cellId = _cartesianCellsArray->InsertNextCell(4);
//                    _cartesianCellsArray->InsertCellPoint(pc - 4);
//                    _cartesianCellsArray->InsertCellPoint(pc - 3);
//                    _cartesianCellsArray->InsertCellPoint(pc - 2);
//                    _cartesianCellsArray->InsertCellPoint(pc - 1);
//
//                    conArray->InsertNextValue(static_cast<double>(con)); // Ensure it works for all numeric types
//                    ++offset;
//                }
//            }
//
//            return true;
//        }

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
        fillScalarFieldCellLevelData3D(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cellTypeArrayAddr,
                                       std::string _conFieldName, std::vector<int> *_typesInvisibeVec,
                                       bool type_indicator_only);

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

        virtual std::vector<int>
        fillCellFieldData3D(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _cellIdArrayAddr,
                            bool extractOuterShellOnly = false);

        virtual bool fillConFieldData3D(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cellTypeArrayAddr,
                                        std::string _conFieldName, std::vector<int> *_typesInvisibeVec,
                                        bool type_indicator_only
        );

        virtual std::vector<int> fillCellFieldGlyphs3D(vtk_obj_addr_int_t centroids_array_addr,
                                                       vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                                                       vtk_obj_addr_int_t cell_type_array_addr,
                                                       std::vector<int> *types_invisibe_vec,
                                                       bool extractOuterShellOnly = false);
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

        virtual std::vector<int> fillScalarFieldGlyphs3D(std::string con_field_name,
                                                         vtk_obj_addr_int_t centroids_array_addr,
                                                         vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                                                         vtk_obj_addr_int_t scalar_value_at_com_addr,
                                                         std::vector<int> *types_invisibe_vec,
                                                         bool extractOuterShellOnly = false);

        virtual std::vector<int> fillScalarFieldCellLevelGlyphs3D(std::string con_field_name,
                                                                  vtk_obj_addr_int_t centroids_array_addr,
                                                                  vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                                                                  vtk_obj_addr_int_t scalar_value_at_com_addr,
                                                                  std::vector<int> *types_invisibe_vec,
                                                                  bool extractOuterShellOnly = false);

        virtual std::vector<int> fillConFieldGlyphs3D(std::string con_field_name,
                                                      vtk_obj_addr_int_t centroids_array_addr,
                                                      vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                                                      vtk_obj_addr_int_t scalar_value_at_com_addr,
                                                      std::vector<int> *types_invisibe_vec,
                                                      bool extractOuterShellOnly = false);

        virtual bool fillLinksField2D(vtk_obj_addr_int_t points_array_addr, 
                                      vtk_obj_addr_int_t lines_array_addr, 
                                      const std::string &plane, 
                                      const int &pos,
                                      const int &margin=1) override;

        virtual bool fillLinksField3D(vtk_obj_addr_int_t points_array_addr, 
                                      vtk_obj_addr_int_t lines_array_addr) override;

        void setVtkObj(void *_vtkObj);

        void setVtkObjInt(long _vtkObjAddr);

        vtkIntArray *produceVtkIntArray();

        int *produceArray(int _size);

        void init(Simulator *_sim);


    private:
        FieldStorage *fsPtr;
        ParallelUtilsOpenMP *pUtils;

        typedef int (FieldExtractor::*type_fcn_ptr_t)(int type);

        FieldExtractor::type_fcn_ptr_t type_fcn_ptr;

    };
};

#endif
