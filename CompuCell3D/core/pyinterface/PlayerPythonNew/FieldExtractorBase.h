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
        /**
         * returns indexes that when applied to permuted (x,y,z) indices (when doing 2D projection)
         * e.g. (x,z,y) for xz projection will return coordinates in the(x, y, z) order
         *      [0, 1, 2][0, 1,2] = [0, 1, 2] - xy projection
         *      [0, 2, 1][0, 2, 1] = [0, 1,2] - xz projection
         *       [1, 2, 0][2, 0, 1] = [0, 1,2] - yz projection
         *
         * @param _plane
         * @return
         */
        std::vector<int> permuted_order_to_xyz(std::string _plane);

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
                 std::string plane, int pos){}
        /**
         *  Fills scalar field 2D glyphs by assigning cells' COM concentration field values to glyph actor
         * @param con_field_name
         * @param centroids_array_addr
         * @param vol_scaling_factors_array_addr
         * @param scalar_value_at_com_addr
         * @param plane
         * @param pos
         */
        virtual void fillConFieldGlyphs2D(
                std::string con_field_name,
                vtk_obj_addr_int_t centroids_array_addr,
                vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                vtk_obj_addr_int_t scalar_value_at_com_addr,
                std::string plane, int pos){}

        /**
         *  Fills scalar field 2D glyphs by assigning cells' COM scalar field values to glyph actor
         * @param con_field_name
         * @param centroids_array_addr
         * @param vol_scaling_factors_array_addr
         * @param scalar_value_at_com_addr
         * @param plane
         * @param pos
         */
        virtual void fillScalarFieldGlyphs2D(
                std::string con_field_name,
                vtk_obj_addr_int_t centroids_array_addr,
                vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                vtk_obj_addr_int_t scalar_value_at_com_addr,
                std::string plane, int pos){}

        /**
         *  Fills scalar field 2D glyphs by assigning cells' COM scalar field cell-level values to glyph actor
         * @param con_field_name
         * @param centroids_array_addr
         * @param vol_scaling_factors_array_addr
         * @param scalar_value_at_com_addr
         * @param plane
         * @param pos
         */
        virtual void fillScalarFieldCellLevelGlyphs2D(
                std::string con_field_name,
                vtk_obj_addr_int_t centroids_array_addr,
                vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                vtk_obj_addr_int_t scalar_value_at_com_addr,
                std::string plane, int pos){}

        /**
         * Fills scalar field glyphs by assigning cells' COM scalar field values to glyph actor
         * @param con_field_name
         * @param centroids_array_addr
         * @param vol_scaling_factors_array_addr
         * @param scalar_value_at_com_addr
         * @param _types_invisibe_vec
         * @param extractOuterShellOnly
         * @return
         */
        virtual std::vector<int> fillScalarFieldGlyphs3D(std::string con_field_name,
                                             vtk_obj_addr_int_t centroids_array_addr,
                                             vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                                             vtk_obj_addr_int_t scalar_value_at_com_addr,
                                             std::vector<int> *_types_invisibe_vec,
                                             bool extractOuterShellOnly = false){return {};}
        /**
         * Fills scalar field glyphs by assigning cells' COM scalar field cell-level values to glyph actor
         * @param con_field_name
         * @param centroids_array_addr
         * @param vol_scaling_factors_array_addr
         * @param scalar_value_at_com_addr
         * @param _types_invisibe_vec
         * @param extractOuterShellOnly
         * @return
         */
        virtual std::vector<int> fillScalarFieldCellLevelGlyphs3D(std::string con_field_name,
                                                         vtk_obj_addr_int_t centroids_array_addr,
                                                         vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                                                         vtk_obj_addr_int_t scalar_value_at_com_addr,
                                                         std::vector<int> *types_invisibe_vec,
                                                         bool extractOuterShellOnly = false){return {};}

        /**
         * Fills concentration field glyphs by assigning cells' COM scalar field values to glyph actor
         * @param _conFieldName
         * @param centroids_array_addr
         * @param vol_scaling_factors_array_addr
         * @param scalar_value_at_com_addr
         * @param _types_invisibe_vec
         * @param extractOuterShellOnly
         * @return
         */
        virtual std::vector<int> fillConFieldGlyphs3D(std::string con_field_name,
                                                                  vtk_obj_addr_int_t centroids_array_addr,
                                                                  vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                                                                  vtk_obj_addr_int_t scalar_value_at_com_addr,
                                                                  std::vector<int> *types_invisibe_vec,
                                                                  bool extractOuterShellOnly = false){return {};}

        virtual bool fillLinksField2D(vtk_obj_addr_int_t points_array_addr, 
                                      vtk_obj_addr_int_t lines_array_addr, 
                                      const std::string &plane,
                                      const int &pos, 
                                      const int &margin=1) { return false; }

        virtual bool fillLinksField3D(vtk_obj_addr_int_t points_array_addr, 
                                      vtk_obj_addr_int_t lines_array_addr) { return false; }


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
         * given list of points - corresponding to a given coordinate (x,y, or z) this fcn
         * computes centroid for this list of coordinate
         * @param point_list
         * @return
         */
        double centroid(const std::list<int> & point_list);

        void setLatticeType(std::string latticeType){this->latticeType=latticeType;}

    protected:
        std::vector<Coordinates3D<double> > hexagonVertices;
        std::vector<Coordinates3D<double> > cartesianVertices;
        std::string latticeType;

        bool isVisible2D(const double &planeCoord, const double &planePos, const double &margin=1.0);

        bool isWithinLattice(double coord, int coordDim, const double &eps=0.01);

        void computeVectorPieceToAdd2D(
                double pt0[2], 
                double pt1[2], 
                const int &clip_coord_idx, 
                const int &other_coord_idx, 
                const int &clip_pos, 
                double (&vector_piece_to_add)[2]
        );

        double otherIntersect2D(
                double pt[2], 
                double vecToAdd[2], 
                int coord_idx_array[2]
        );

        void computeClippedSegment2D(double pt0[2], double pt1[2], int fieldDimOrdered[3], double (&vecToAdd)[2]);

        void computeClippedSegment3D(double pt0[3], double pt1[3], int fieldDim[3], double (&vecToAdd)[3]);

        bool linksPos2D(
                double pt0[3], 
                double pt1[3], 
                int fieldDimOrdered[3], 
                int dimOrder[3],
                double (&link0_begin)[2], 
                double (&link0_end)[2], 
                double (&link1_begin)[2], 
                double (&link1_end)[2]
        );

        bool linksPos3D(
                double pt0[3], 
                double pt1[3], 
                int fieldDim[3], 
                double (&link0_begin)[3], 
                double (&link0_end)[3], 
                double (&link1_begin)[3], 
                double (&link1_end)[3]
        );

        void vizLinks2D(
                double pt0[3], 
                double pt1[3], 
                vtk_obj_addr_int_t points_addr, 
                vtk_obj_addr_int_t lines_addr, 
                int fieldDimOrdered[3], 
                int dimOrder[3], 
                const double &planePos, 
                const double &margin, 
                int &ptCounter
        );

        void vizLinks3D(
                double pt0[3], 
                double pt1[3], 
                vtk_obj_addr_int_t points_addr, 
                vtk_obj_addr_int_t lines_addr, 
                int fieldDim[3], 
                int &ptCounter
        );

    };
};

#endif
