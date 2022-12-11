#ifndef FIELDEXTRACTORBASE_H
#define FIELDEXTRACTORBASE_H

#include <vector>
#include <map>
#include <list>
#include <algorithm>
#include <string>
#include <Utils/Coordinates3D.h>
#include "FieldExtractorDLLSpecifier.h"
#include "FieldExtractorTypes.h"

//Notice one can speed up filling up of the Hex lattice data by allocating e.g. hexPOints ot cellType arrays
//instead of inserting values. Inserting causes reallocations and this slows down the task completion

namespace CompuCell3D {

    class FIELDEXTRACTOR_EXPORT  FieldExtractorBase {
    public:

//        typedef int (FieldExtractorBase::*type_fcn_ptr_t)(int type);
//        FieldExtractorBase::type_fcn_ptr_t type_fcn_ptr;

        FieldExtractorBase();

        virtual ~FieldExtractorBase();

        std::vector<int> pointOrder(std::string _plane);

        std::vector<int> dimOrder(std::string _plane);

        Coordinates3D<double> HexCoordXY(unsigned int x, unsigned int y, unsigned int z);

        virtual void fillCellFieldData2D(vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _plane, int _pos);

        virtual void
        fillBorderData2D(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr, std::string _plane,
                         int _pos) {}

        virtual void
        fillCentroidData2D(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr, std::string _plane,
                           int _pos) {}

        virtual void
        fillCellFieldData2DHex(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _hexCellsArrayAddr,
                               vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane, int _pos) {}

        virtual void
        fillBorderData2DHex(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr, std::string _plane,
                            int _pos) {}

        virtual void fillClusterBorderData2D(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr,
                                             std::string _plane, int _pos) {}

        virtual void fillClusterBorderData2DHex(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr,
                                                std::string _plane, int _pos) {}

        virtual bool fillConFieldData2D(vtk_obj_addr_int_t _conArrayAddr, std::string _conFieldName, std::string _plane,
                                        int _pos) { return false; }

        virtual bool fillConFieldData2DHex(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _hexCellsArrayAddr,
                                           vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName,
                                           std::string _plane, int _pos) { return false; }

        virtual bool
        fillConFieldData2DCartesian(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cartesianCellsArrayAddr,
                                    vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName, std::string _plane,
                                    int _pos) { return false; }

        virtual bool
        fillScalarFieldData2D(vtk_obj_addr_int_t _conArrayAddr, std::string _conFieldName, std::string _plane,
                              int _pos) { return false; }

        virtual bool fillScalarFieldData2DHex(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _hexCellsArrayAddr,
                                              vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName,
                                              std::string _plane, int _pos) { return false; }

        virtual bool
        fillScalarFieldData2DCartesian(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cartesianCellsArrayAddr,
                                       vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName,
                                       std::string _plane, int _pos) { return false; }

        virtual bool
        fillScalarFieldCellLevelData2D(vtk_obj_addr_int_t _conArrayAddr, std::string _conFieldName, std::string _plane,
                                       int _pos) { return false; }

        virtual bool
        fillScalarFieldCellLevelData2DHex(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _hexCellsArrayAddr,
                                          vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName,
                                          std::string _plane, int _pos) { return false; }

        virtual bool fillScalarFieldCellLevelData2DCartesian(vtk_obj_addr_int_t _conArrayAddr,
                                                             vtk_obj_addr_int_t _cartesianCellsArrayAddr,
                                                             vtk_obj_addr_int_t _pointsArrayAddr,
                                                             std::string _conFieldName, std::string _plane,
                                                             int _pos) { return false; }

        virtual bool
        fillVectorFieldData2D(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr,
                              std::string _fieldName, std::string _plane, int _pos) { return false; }

        virtual bool
        fillVectorFieldData2DHex(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr,
                                 std::string _fieldName, std::string _plane, int _pos) { return false; }

        virtual bool
        fillVectorFieldData3D(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr,
                              std::string _fieldName) { return false; }

        virtual bool
        fillVectorFieldCellLevelData2D(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr,
                                       std::string _fieldName, std::string _plane, int _pos) { return false; }

        virtual bool fillVectorFieldCellLevelData2DHex(vtk_obj_addr_int_t _pointsArrayIntAddr,
                                                       vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName,
                                                       std::string _plane, int _pos) { return false; }

        virtual bool
        fillVectorFieldCellLevelData3D(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr,
                                       std::string _fieldName) { return false; }

        virtual bool fillScalarFieldData3D(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cellTypeArrayAddr,
                                           std::string _conFieldName,
                                           std::vector<int> *_typesInvisibeVec,
                                           bool type_indicator_only = false) { return false; }

        virtual bool
        fillScalarFieldCellLevelData3D(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cellTypeArrayAddr,
                                       std::string _conFieldName, std::vector<int> *_typesInvisibeVec,
                                       bool type_indicator_only = false) { return false; }

        virtual std::vector<int>
        fillCellFieldData3D(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _cellIdArrayAddr,
                            bool extractOuterShellOnly = false) { return std::vector<int>(); }

        virtual bool fillConFieldData3D(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cellTypeArrayAddr,
                                        std::string _conFieldName, std::vector<int> *_typesInvisibeVec,
                                        bool type_indicator_only = false
        ) { return false; }

        virtual std::vector<int> fillCellFieldGlyphs3D(vtk_obj_addr_int_t centroids_array_addr,
                                                       vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                                                       vtk_obj_addr_int_t cell_type_array_addr,
                                                       std::vector<int> *_types_invisibe_vec,
                                                       bool extractOuterShellOnly = false) { return std::vector<int>(); }

        /**
         * fills glyph data for cell field in 2D projection
         * @param centroids_array_addr
         * @param vol_scaling_factors_array_addr
         * @param cell_type_array_addr
         * @param plane
         * @param pos
         */
        virtual void fillCellFieldGlyphs2D(
                vtk_obj_addr_int_t centroids_array_addr,
                vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                vtk_obj_addr_int_t cell_type_array_addr,
                 std::string plane, int _pos){}

        /**
         * returns 0 for medium and 1 for all non medium types
         * @param type - cell type_id
         * @return
         */
        int type_indicator(int type) { return std::min(1, type); }


        /**
         * returns 0 for medium and type_id for all other types
         * @param type - cell type_id
         * @return
         */
        int type_value(int type) { return type; }

        /**
         * fiven list of points - corresponding to a given coordinate (x,y, or z) this fcn
         * computes centroid for this list of coordinate
         * @param point_list
         * @return
         */
        double centroid(const std::list<int> & point_list);

    protected:
        std::vector<Coordinates3D<double> > hexagonVertices;
        std::vector<Coordinates3D<double> > cartesianVertices;

    };
};

#endif
