

#include <iostream>
#include <sstream>
#include <CompuCell3D/Field3D/Dim3D.h>

#include <PublicUtilities/NumericalUtils.h>
#include <Utils/Coordinates3D.h>
#include <algorithm>
#include <cmath>
#include <set>

#include <vtkPythonUtil.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <numeric>

using namespace std;
using namespace CompuCell3D;


#include "FieldExtractorBase.h"


FieldExtractorBase::FieldExtractorBase() {

    double sqrt_3_3 = sqrt(3.0) / 3.0;
    hexagonVertices.push_back(Coordinates3D<double>(0, sqrt_3_3, 0.0));
    hexagonVertices.push_back(Coordinates3D<double>(0.5, 0.5 * sqrt_3_3, 0.0));
    hexagonVertices.push_back(Coordinates3D<double>(0.5, -0.5 * sqrt_3_3, 0.0));
    hexagonVertices.push_back(Coordinates3D<double>(0., -sqrt_3_3, 0.0));
    hexagonVertices.push_back(Coordinates3D<double>(-0.5, -0.5 * sqrt_3_3, 0.0));
    hexagonVertices.push_back(Coordinates3D<double>(-0.5, 0.5 * sqrt_3_3, 0.0));

    cartesianVertices.push_back(Coordinates3D<double>(0.0, 0.0, 0.0));
    cartesianVertices.push_back(Coordinates3D<double>(0.0, 1.0, 0.0));
    cartesianVertices.push_back(Coordinates3D<double>(1.0, 1.0, 0.0));
    cartesianVertices.push_back(Coordinates3D<double>(1.0, 0.0, 0.0));


}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldExtractorBase::~FieldExtractorBase() {

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<int> FieldExtractorBase::pointOrder(std::string _plane) {
    for (int i = 0; i < _plane.size(); ++i) {
        _plane[i] = tolower(_plane[i]);
    }

    std::vector<int> order(3, 0);
    order[0] = 0;
    order[1] = 1;
    order[2] = 2;
    if (_plane == "xy") {
        order[0] = 0;
        order[1] = 1;
        order[2] = 2;
    } else if (_plane == "xz") {
        order[0] = 0;
        order[1] = 2;
        order[2] = 1;


    } else if (_plane == "yz") {
        order[0] = 2;
        order[1] = 0;
        order[2] = 1;


    }
    return order;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<int> FieldExtractorBase::dimOrder(std::string _plane) {
    for (int i = 0; i < _plane.size(); ++i) {
        _plane[i] = tolower(_plane[i]);
    }

    std::vector<int> order(3, 0);
    order[0] = 0;
    order[1] = 1;
    order[2] = 2;
    if (_plane == "xy") {
        order[0] = 0;
        order[1] = 1;
        order[2] = 2;
    } else if (_plane == "xz") {
        order[0] = 0;
        order[1] = 2;
        order[2] = 1;


    } else if (_plane == "yz") {
        order[0] = 1;
        order[1] = 2;
        order[2] = 0;


    }
    return order;
}

std::vector<int> FieldExtractorBase::permuted_order_to_xyz(std::string _plane) {
    for (int i = 0; i < _plane.size(); ++i) {
        _plane[i] = tolower(_plane[i]);
    }

    // [0, 1, 2][0, 1,2] = [0, 1, 2] - xy projection
    // [0, 2, 1][0, 2, 1] = [0, 1,2] - xz projection
    // [1, 2, 0][2, 0, 1] = [0, 1,2 ] - yz projection

    std::vector<int> order(3, 0);
    order[0] = 0;
    order[1] = 1;
    order[2] = 2;
    if (_plane == "xy") {
        order[0] = 0;
        order[1] = 1;
        order[2] = 2;
    } else if (_plane == "xz") {
        order[0] = 0;
        order[1] = 2;
        order[2] = 1;


    } else if (_plane == "yz") {
        order[0] = 2;
        order[1] = 0;
        order[2] = 1;


    }
    return order;
}


Coordinates3D<double> FieldExtractorBase::HexCoordXY(unsigned int x, unsigned int y, unsigned int z) {
    //coppied from BoundaryStrategy.cpp HexCoord fcn
    if ((z % 3) == 1) {//odd z e.g. z=1

        if (y % 2)
            return Coordinates3D<double>(x + 0.5, sqrt(3.0) / 2.0 * (y + 2.0 / 6.0), z * sqrt(6.0) / 3.0);
        else//even
            return Coordinates3D<double>(x, sqrt(3.0) / 2.0 * (y + 2.0 / 6.0), z * sqrt(6.0) / 3.0);


    } else if ((z % 3) == 2) { //e.g. z=2


        if (y % 2)
            return Coordinates3D<double>(x + 0.5, sqrt(3.0) / 2.0 * (y - 2.0 / 6.0), z * sqrt(6.0) / 3.0);
        else//even
            return Coordinates3D<double>(x, sqrt(3.0) / 2.0 * (y - 2.0 / 6.0), z * sqrt(6.0) / 3.0);


    } else {//z divible by 3 - includes z=0
        if (y % 2)
            return Coordinates3D<double>(x, sqrt(3.0) / 2.0 * y, z * sqrt(6.0) / 3.0);
        else//even
            return Coordinates3D<double>(x + 0.5, sqrt(3.0) / 2.0 * y, z * sqrt(6.0) / 3.0);
    }


}

void FieldExtractorBase::fillCellFieldData2D(vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _plane, int _pos) {}

double FieldExtractorBase::centroid(const list<int> &point_list) {
    auto res = std::accumulate(point_list.begin(), point_list.end(), 0.0);
    return res / point_list.size();
}

bool FieldExtractorBase::isVisible2D(const double &planeCoord, const double &planePos, const double &margin) {
    return planeCoord - margin <= planePos && planePos < planeCoord + margin;
}

bool FieldExtractorBase::isWithinLattice(double coord, int coordDim, const double &eps) {
    return coord >= -eps && coord <= coordDim + eps;
}

void FieldExtractorBase::computeVectorPieceToAdd2D(
    double pt0[2], 
    double pt1[2], 
    const int &clip_coord_idx, 
    const int &other_coord_idx, 
    const int &clip_pos, 
    double (&vector_piece_to_add)[2]
) {
    double ratio = (pt1[clip_coord_idx] - clip_pos) / (pt1[clip_coord_idx] - pt0[clip_coord_idx]);
    vector_piece_to_add[clip_coord_idx] = clip_pos - pt0[clip_coord_idx];
    vector_piece_to_add[other_coord_idx] = (1 - ratio) * (pt1[other_coord_idx] - pt0[other_coord_idx]);
}

double FieldExtractorBase::otherIntersect2D(
    double pt[2], 
    double vecToAdd[2], 
    int coord_idx_array[2]
) {
    int other_coord_idx = coord_idx_array[1];
    return pt[other_coord_idx] + vecToAdd[other_coord_idx];
}

void FieldExtractorBase::computeClippedSegment2D(double pt0[2], double pt1[2], int fieldDimOrdered[3], double (&vecToAdd)[2]) {
    vecToAdd[0] = pt1[0] - pt0[0];
    vecToAdd[1] = pt1[1] - pt0[1];

    for(int i = 0; i < 2; i++) 
        if(!isWithinLattice(pt1[i], fieldDimOrdered[i])) {
            int pos = pt1[i] < 0 ? 0 : fieldDimOrdered[i];
            int coord_idx_array[] = {1, 1};
            coord_idx_array[i] = 0;
            
            computeVectorPieceToAdd2D(pt0, pt1, coord_idx_array[0], coord_idx_array[1], pos, vecToAdd);
            double other = otherIntersect2D(pt0, vecToAdd, coord_idx_array);
            if(isWithinLattice(other, fieldDimOrdered[coord_idx_array[1]])) 
                break;
        }
}

void FieldExtractorBase::computeClippedSegment3D(double pt0[3], double pt1[3], int fieldDim[3], double (&vecToAdd)[3]) {
    vecToAdd[0] = pt1[0] - pt0[0];
    vecToAdd[1] = pt1[1] - pt0[1];
    vecToAdd[2] = pt1[2] - pt0[2];

    for(int i = 0; i < 3; i++) 
        if(!isWithinLattice(pt1[i], fieldDim[i])) {
            int pos = pt1[i] < 0 ? 0 : fieldDim[i];
            vecToAdd[i] = pos - pt0[i];
        }
}

bool FieldExtractorBase::linksPos2D(
    double pt0[3], 
    double pt1[3], 
    int fieldDimOrdered[3], 
    int dimOrder[3], 
    double (&link0_begin)[2], 
    double (&link0_end)[2], 
    double (&link1_begin)[2], 
    double (&link1_end)[2]
) {
    link0_begin[0] = pt0[dimOrder[0]];
    link0_begin[1] = pt0[dimOrder[1]];
    link1_begin[0] = pt1[dimOrder[0]];
    link1_begin[1] = pt1[dimOrder[1]];

    Coordinates3D<double> invDistCoords = distanceVectorCoordinatesInvariant(
        {link1_begin[0], link1_begin[1], 0}, 
        {link0_begin[0], link0_begin[1], 0}, 
        {static_cast<short>(fieldDimOrdered[0]), static_cast<short>(fieldDimOrdered[1]), 1}
    );

    link0_end[0] = link0_begin[0] + invDistCoords.x;
    link0_end[1] = link0_begin[1] + invDistCoords.y;
    if(isWithinLattice(link0_end[0], fieldDimOrdered[0]) && isWithinLattice(link0_end[1], fieldDimOrdered[1])) 
        return false;

    link1_begin[0] = link0_end[0];
    link1_begin[1] = link0_end[1];

    double vecToAdd[2];
    computeClippedSegment2D(link0_begin, link0_end, fieldDimOrdered, vecToAdd);
    link0_end[0] = link0_begin[0] + vecToAdd[0];
    link0_end[1] = link0_begin[1] + vecToAdd[1];

    link1_end[0] = link1_begin[0] + vecToAdd[0] - invDistCoords.x;
    link1_end[1] = link1_begin[1] + vecToAdd[1] - invDistCoords.y;
    computeClippedSegment2D(link1_begin, link1_end, fieldDimOrdered, vecToAdd);
    link1_end[0] = link1_begin[0] + vecToAdd[0];
    link1_end[1] = link1_begin[1] + vecToAdd[1];

    return true;
}

bool FieldExtractorBase::linksPos3D(
    double pt0[3], 
    double pt1[3], 
    int fieldDim[3], 
    double (&link0_begin)[3], 
    double (&link0_end)[3], 
    double (&link1_begin)[3], 
    double (&link1_end)[3]
) {
    link0_begin[0] = pt0[0];
    link0_begin[1] = pt0[1];
    link0_begin[2] = pt0[2];
    link1_begin[0] = pt1[0];
    link1_begin[1] = pt1[1];
    link1_begin[2] = pt1[2];

    Coordinates3D<double> invDistCoords = distanceVectorCoordinatesInvariant(
        {link1_begin[0], link1_begin[1], link1_begin[2]}, 
        {link0_begin[0], link0_begin[1], link0_begin[2]}, 
        {static_cast<short>(fieldDim[0]), static_cast<short>(fieldDim[1]), static_cast<short>(fieldDim[2])}
    );

    link0_end[0] = link0_begin[0] + invDistCoords.x;
    link0_end[1] = link0_begin[1] + invDistCoords.y;
    link0_end[2] = link0_begin[2] + invDistCoords.z;
    if(isWithinLattice(link0_end[0], fieldDim[0]) && isWithinLattice(link0_end[1], fieldDim[1]) && isWithinLattice(link0_end[2], fieldDim[2])) 
        return false;

    link1_begin[0] = link0_end[0];
    link1_begin[1] = link0_end[1];
    link1_begin[2] = link0_end[2];

    double vecToAdd[3];
    computeClippedSegment3D(link0_begin, link0_end, fieldDim, vecToAdd);
    link0_end[0] = link0_begin[0] + vecToAdd[0];
    link0_end[1] = link0_begin[1] + vecToAdd[1];
    link0_end[2] = link0_begin[2] + vecToAdd[2];
    computeClippedSegment3D(link1_begin, link1_end, fieldDim, vecToAdd);
    link1_end[0] = link1_begin[0] + vecToAdd[0] - invDistCoords.x;
    link1_end[1] = link1_begin[1] + vecToAdd[1] - invDistCoords.y;
    link1_end[2] = link1_begin[2] + vecToAdd[2] - invDistCoords.z;

    return true;
}

void FieldExtractorBase::vizLinks2D(
    double pt0[3], 
    double pt1[3], 
    vtk_obj_addr_int_t points_addr, 
    vtk_obj_addr_int_t lines_addr, 
    int fieldDimOrdered[3], 
    int dimOrder[3], 
    const double &planePos, 
    const double &margin, 
    int &ptCounter
) {
    vtkPoints *points = (vtkPoints*) points_addr;
    vtkCellArray *lines = (vtkCellArray*) lines_addr;

    bool viz0 = isVisible2D(pt0[dimOrder[2]], planePos, margin);
    bool viz1 = isVisible2D(pt1[dimOrder[2]], planePos, margin);

    if(!(viz0 || viz1)) 
        return;

    double linkBegin0[2], linkEnd0[2], linkBegin1[2], linkEnd1[2];
    const bool clipped = linksPos2D(pt0, pt1, fieldDimOrdered, dimOrder, linkBegin0, linkEnd0, linkBegin1, linkEnd1);

    if(viz0) {
        points->InsertNextPoint(linkBegin0[0], linkBegin0[1], 0);
        points->InsertNextPoint(linkEnd0[0], linkEnd0[1], 0);

        lines->InsertNextCell(2);
        lines->InsertCellPoint(ptCounter);
        lines->InsertCellPoint(ptCounter + 1);
        ptCounter += 2;
    }
    if(viz1 && clipped) {
        points->InsertNextPoint(linkBegin1[0], linkBegin1[1], 0);
        points->InsertNextPoint(linkEnd1[0], linkEnd1[1], 0);

        lines->InsertNextCell(2);
        lines->InsertCellPoint(ptCounter);
        lines->InsertCellPoint(ptCounter + 1);
        ptCounter += 2;
    }
}

void FieldExtractorBase::vizLinks3D(
    double pt0[3], 
    double pt1[3], 
    vtk_obj_addr_int_t points_addr, 
    vtk_obj_addr_int_t lines_addr, 
    int fieldDim[3], 
    int &ptCounter
) {
    vtkPoints *points = (vtkPoints*) points_addr;
    vtkCellArray *lines = (vtkCellArray*) lines_addr;

    double linkBegin0[3], linkEnd0[3], linkBegin1[3], linkEnd1[3];
    const bool clipped = linksPos3D(pt0, pt1, fieldDim, linkBegin0, linkEnd0, linkBegin1, linkEnd1);

    points->InsertNextPoint(linkBegin0[0], linkBegin0[1], linkBegin0[2]);
    points->InsertNextPoint(linkEnd0[0], linkEnd0[1], linkEnd0[2]);

    lines->InsertNextCell(2);
    lines->InsertCellPoint(ptCounter);
    lines->InsertCellPoint(ptCounter + 1);
    ptCounter += 2;
    if(clipped) {
        points->InsertNextPoint(linkBegin1[0], linkBegin1[1], linkBegin1[2]);
        points->InsertNextPoint(linkEnd1[0], linkEnd1[1], linkEnd1[2]);

        lines->InsertNextCell(2);
        lines->InsertCellPoint(ptCounter);
        lines->InsertCellPoint(ptCounter + 1);
        ptCounter += 2;
    }
}
