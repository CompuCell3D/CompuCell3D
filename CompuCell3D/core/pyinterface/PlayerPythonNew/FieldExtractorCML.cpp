

#include <iostream>
#include <CompuCell3D/Field3D/Dim3D.h>
#include <Utils/Coordinates3D.h>
#include <vtkIntArray.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkType.h>
#include <vtkCharArray.h>
#include <vtkLongArray.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkStructuredPoints.h>
#include <vtkStructuredPointsReader.h>
#include <algorithm>
#include <cmath>
#include <set>

#include <vtkPythonUtil.h>
#include <unordered_map>
#include <unordered_set>

using namespace std;
using namespace CompuCell3D;


#include "FieldExtractorCML.h"
#include "FieldWriterCML.h"


FieldExtractorCML::FieldExtractorCML() : lds(0), zDimFactor(0), yDimFactor(0) {

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldExtractorCML::~FieldExtractorCML() {

}

void FieldExtractorCML::setSimulationData(vtk_obj_addr_int_t _structuredPointsAddr) {

    lds = (vtkStructuredPoints *) _structuredPointsAddr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Dim3D FieldExtractorCML::getFieldDim() {
    return fieldDim;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FieldExtractorCML::setFieldDim(Dim3D _dim) {
    fieldDim = _dim;
    zDimFactor = fieldDim.x * fieldDim.y;
    yDimFactor = fieldDim.x;
}

long FieldExtractorCML::pointIndex(short _x, short _y, short _z) {
    return zDimFactor * _z + yDimFactor * _y + _x;
}

long FieldExtractorCML::indexPoint3D(Point3D _pt) {
    return zDimFactor * _pt.z + yDimFactor * _pt.y + _pt.x;
}

void FieldExtractorCML::fillCentroidData2D(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr,
                                           std::string _plane, int _pos) {

}

void FieldExtractorCML::fillCellFieldData2D(vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _plane, int _pos) {


    vtkIntArray *_cellTypeArray = (vtkIntArray *) _cellTypeArrayAddr;
    vtkCharArray *typeArrayRead = (vtkCharArray *) lds->GetPointData()->GetArray("CellType");

    vector<int> fieldDimVec(3, 0);
    fieldDimVec[0] = fieldDim.x;
    fieldDimVec[1] = fieldDim.y;
    fieldDimVec[2] = fieldDim.z;

    vector<int> pointOrderVec = pointOrder(_plane);
    vector<int> dimOrderVec = dimOrder(_plane);

    vector<int> dim(3, 0);
    dim[0] = fieldDimVec[dimOrderVec[0]];
    dim[1] = fieldDimVec[dimOrderVec[1]];
    dim[2] = fieldDimVec[dimOrderVec[2]];

    _cellTypeArray->SetNumberOfValues((dim[1] + 2) * (dim[0] + 1));

    //For some reasons the points x=0 are eaten up (don't know why).
    //So we just populate empty cellIds.
    int offset = 0;
    for (int i = 0; i < dim[0] + 1; ++i) {
        _cellTypeArray->SetValue(offset, 0);
        ++offset;
    }

    Point3D pt;
    vector<int> ptVec(3, 0);

    int type;
    //long index;
    //when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
    for (int j = 0; j < dim[1] + 1; ++j)
        for (int i = 0; i < dim[0] + 1; ++i) {
            ptVec[0] = i;
            ptVec[1] = j;
            ptVec[2] = _pos;

            pt.x = ptVec[pointOrderVec[0]];
            pt.y = ptVec[pointOrderVec[1]];
            pt.z = ptVec[pointOrderVec[2]];

            if (i >= dim[0] || j >= dim[1]) {
                _cellTypeArray->SetValue(offset, 0);
            } else {
                _cellTypeArray->SetValue(offset, typeArrayRead->GetValue(indexPoint3D(pt)));
            }
            ++offset;
        }
}

void FieldExtractorCML::fillCellFieldData2DCartesian(vtk_obj_addr_int_t _cellTypeArrayAddr,
                                                     vtk_obj_addr_int_t _cellsArrayAddr,
                                                     vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane,
                                                     int _pos) {

    vtkIntArray *_cellTypeArray = (vtkIntArray *) _cellTypeArrayAddr;
    vtkPoints *_pointsArray = (vtkPoints *) _pointsArrayAddr;
    vtkCellArray *_cellsArray = (vtkCellArray *) _cellsArrayAddr;
    vtkCharArray *typeArrayRead = (vtkCharArray *) lds->GetPointData()->GetArray("CellType");

    //Field3D<CellG*> * cellFieldG = potts->getCellFieldG();
    //Dim3D fieldDim = cellFieldG->getDim();

    vector<int> fieldDimVec(3, 0);
    fieldDimVec[0] = fieldDim.x;
    fieldDimVec[1] = fieldDim.y;
    fieldDimVec[2] = fieldDim.z;

    vector<int> pointOrderVec = pointOrder(_plane);
    vector<int> dimOrderVec = dimOrder(_plane);

    vector<int> dim(3, 0);
    dim[0] = fieldDimVec[dimOrderVec[0]];
    dim[1] = fieldDimVec[dimOrderVec[1]];
    dim[2] = fieldDimVec[dimOrderVec[2]];

    int offset = 0;

    Point3D pt;
    vector<int> ptVec(3, 0);
    //CellG* cell;
    //int type;
    long pc = 0;

    char cellType;

    //when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned


    for (int j = 0; j < dim[1]; ++j)
        for (int i = 0; i < dim[0]; ++i) {
            ptVec[0] = i;
            ptVec[1] = j;
            ptVec[2] = _pos;

            pt.x = ptVec[pointOrderVec[0]];
            pt.y = ptVec[pointOrderVec[1]];
            pt.z = ptVec[pointOrderVec[2]];

            //cell = cellFieldG->get(pt);
            cellType = typeArrayRead->GetValue(indexPoint3D(pt));
            //if (!cell) {
            //    type = 0;
            //    continue;
            //}
            //else {
            //    type = cell->type;
            //}


            Coordinates3D<double> coords(ptVec[0], ptVec[1],
                                         0); // notice that we are drawing pixels from other planes on a xy plan so we use ptVec instead of pt. pt is absolute position of the point ptVec is for projection purposes

            for (int idx = 0; idx < 4; ++idx) {
                Coordinates3D<double> cartesianVertex = cartesianVertices[idx] + coords;
                _pointsArray->InsertNextPoint(cartesianVertex.x, cartesianVertex.y, 0.0);
            }

            pc += 4;
            vtkIdType cellId = _cellsArray->InsertNextCell(4);
            _cellsArray->InsertCellPoint(pc - 4);
            _cellsArray->InsertCellPoint(pc - 3);
            _cellsArray->InsertCellPoint(pc - 2);
            _cellsArray->InsertCellPoint(pc - 1);

            _cellTypeArray->InsertNextValue(cellType);
            ++offset;
        }


}


void
FieldExtractorCML::fillCellFieldData2DHex(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _hexCellsArrayAddr,
                                          vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane, int _pos) {
    vtkIntArray *_cellTypeArray = (vtkIntArray *) _cellTypeArrayAddr;
    vtkPoints *_pointsArray = (vtkPoints *) _pointsArrayAddr;
    vtkCellArray *_hexCellsArray = (vtkCellArray *) _hexCellsArrayAddr;

    vtkCharArray *typeArrayRead = (vtkCharArray *) lds->GetPointData()->GetArray("CellType");

    vector<int> fieldDimVec(3, 0);
    fieldDimVec[0] = fieldDim.x;
    fieldDimVec[1] = fieldDim.y;
    fieldDimVec[2] = fieldDim.z;

    vector<int> pointOrderVec = pointOrder(_plane);
    vector<int> dimOrderVec = dimOrder(_plane);

    vector<int> dim(3, 0);
    dim[0] = fieldDimVec[dimOrderVec[0]];
    dim[1] = fieldDimVec[dimOrderVec[1]];
    dim[2] = fieldDimVec[dimOrderVec[2]];

    int offset = 0;

    ////For some reasons the points x=0 are eaten up (don't know why).
    ////So we just populate empty cellIds.

    //for (int i = 0 ; i< dim[0]+1 ;++i){
    //	_cellTypeArray->SetValue(offset, 0);
    //	++offset;
    //}

    Point3D pt;
    vector<int> ptVec(3, 0);

    char cellType;

    long pc = 0;
    //when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
    for (int j = 0; j < dim[1]; ++j)
        for (int i = 0; i < dim[0]; ++i) {
            ptVec[0] = i;
            ptVec[1] = j;
            ptVec[2] = _pos;

            pt.x = ptVec[pointOrderVec[0]];
            pt.y = ptVec[pointOrderVec[1]];
            pt.z = ptVec[pointOrderVec[2]];

            cellType = typeArrayRead->GetValue(indexPoint3D(pt));

            Coordinates3D<double> hexCoords = HexCoordXY(pt.x, pt.y, pt.z);
            for (int idx = 0; idx < 6; ++idx) {
                Coordinates3D<double> hexagonVertex = hexagonVertices[idx] + hexCoords;
                _pointsArray->InsertNextPoint(hexagonVertex.x, hexagonVertex.y, 0.0);
            }
            pc += 6;
            vtkIdType cellId = _hexCellsArray->InsertNextCell(6);
            _hexCellsArray->InsertCellPoint(pc - 6);
            _hexCellsArray->InsertCellPoint(pc - 5);
            _hexCellsArray->InsertCellPoint(pc - 4);
            _hexCellsArray->InsertCellPoint(pc - 3);
            _hexCellsArray->InsertCellPoint(pc - 2);
            _hexCellsArray->InsertCellPoint(pc - 1);

            _cellTypeArray->InsertNextValue(cellType);

            ++offset;
        }
}

void FieldExtractorCML::fillBorder2D(const char *arrayName, vtk_obj_addr_int_t _pointArrayAddr,
                                     vtk_obj_addr_int_t _linesArrayAddr, std::string _plane, int _pos) {

//	vtkLongArray *idArray=(vtkLongArray *)lds->GetPointData()->GetArray("CellId");
    vtkLongArray *idArray = (vtkLongArray *) lds->GetPointData()->GetArray(arrayName);

    vtkPoints *points = (vtkPoints *) _pointArrayAddr;
    vtkCellArray *lines = (vtkCellArray *) _linesArrayAddr;

    vector<int> fieldDimVec(3, 0);
    fieldDimVec[0] = fieldDim.x;
    fieldDimVec[1] = fieldDim.y;
    fieldDimVec[2] = fieldDim.z;

    vector<int> pointOrderVec = pointOrder(_plane);
    vector<int> dimOrderVec = dimOrder(_plane);

    vector<int> dim(3, 0);
    dim[0] = fieldDimVec[dimOrderVec[0]];
    dim[1] = fieldDimVec[dimOrderVec[1]];
    dim[2] = fieldDimVec[dimOrderVec[2]];

    Point3D pt;
    vector<int> ptVec(3, 0);
    Point3D ptN;
    vector<int> ptNVec(3, 0);

    long idxPt;

    int k = 0;
    int pc = 0;

    for (int i = 0; i < dim[0]; ++i)
        for (int j = 0; j < dim[1]; ++j) {
            ptVec[0] = i;
            ptVec[1] = j;
            ptVec[2] = _pos;

            pt.x = ptVec[pointOrderVec[0]];
            pt.y = ptVec[pointOrderVec[1]];
            pt.z = ptVec[pointOrderVec[2]];

            idxPt = indexPoint3D(pt);

            if (i > 0 && j < dim[1]) {
                ptNVec[0] = i - 1;
                ptNVec[1] = j;
                ptNVec[2] = _pos;
                ptN.x = ptNVec[pointOrderVec[0]];
                ptN.y = ptNVec[pointOrderVec[1]];
                ptN.z = ptNVec[pointOrderVec[2]];
                if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                    points->InsertNextPoint(i, j, 0);
                    points->InsertNextPoint(i, j + 1, 0);
                    pc += 2;
                    lines->InsertNextCell(2);
                    lines->InsertCellPoint(pc - 2);
                    lines->InsertCellPoint(pc - 1);
                }
            }
            if (j > 0 && i < dim[0]) {
                ptNVec[0] = i;
                ptNVec[1] = j - 1;
                ptNVec[2] = _pos;
                ptN.x = ptNVec[pointOrderVec[0]];
                ptN.y = ptNVec[pointOrderVec[1]];
                ptN.z = ptNVec[pointOrderVec[2]];
                if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                    points->InsertNextPoint(i, j, 0);
                    points->InsertNextPoint(i + 1, j, 0);
                    pc += 2;
                    lines->InsertNextCell(2);
                    lines->InsertCellPoint(pc - 2);
                    lines->InsertCellPoint(pc - 1);
                }
            }
            if (i < dim[0] && j < dim[1]) {
                ptNVec[0] = i + 1;
                ptNVec[1] = j;
                ptNVec[2] = _pos;
                ptN.x = ptNVec[pointOrderVec[0]];
                ptN.y = ptNVec[pointOrderVec[1]];
                ptN.z = ptNVec[pointOrderVec[2]];
                if (ptNVec[0] >= dim[0] || idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                    points->InsertNextPoint(i + 1, j, 0);
                    points->InsertNextPoint(i + 1, j + 1, 0);
                    pc += 2;
                    lines->InsertNextCell(2);
                    lines->InsertCellPoint(pc - 2);
                    lines->InsertCellPoint(pc - 1);
                }
            }
            if (i < dim[0] && j < dim[1]) {
                ptNVec[0] = i;
                ptNVec[1] = j + 1;
                ptNVec[2] = _pos;
                ptN.x = ptNVec[pointOrderVec[0]];
                ptN.y = ptNVec[pointOrderVec[1]];
                ptN.z = ptNVec[pointOrderVec[2]];
                if (ptNVec[1] >= dim[1] || idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                    points->InsertNextPoint(i, j + 1, 0);
                    points->InsertNextPoint(i + 1, j + 1, 0);
                    pc += 2;
                    lines->InsertNextCell(2);
                    lines->InsertCellPoint(pc - 2);
                    lines->InsertCellPoint(pc - 1);
                }
            }
        }
}

void FieldExtractorCML::fillBorderData2D(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr,
                                         std::string _plane, int _pos) {
    fillBorder2D("CellId", _pointArrayAddr, _linesArrayAddr, _plane, _pos);
}

void FieldExtractorCML::fillClusterBorderData2D(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr,
                                                std::string _plane, int _pos) {
    fillBorder2D("ClusterId", _pointArrayAddr, _linesArrayAddr, _plane, _pos);
}

void FieldExtractorCML::fillBorder2DHex(const char *arrayName, vtk_obj_addr_int_t _pointArrayAddr,
                                        vtk_obj_addr_int_t _linesArrayAddr, std::string _plane, int _pos) {

    vtkPoints *points = (vtkPoints *) _pointArrayAddr;
    vtkCellArray *lines = (vtkCellArray *) _linesArrayAddr;

//	vtkLongArray *idArray=(vtkLongArray *)lds->GetPointData()->GetArray("CellId");
    vtkLongArray *idArray = (vtkLongArray *) lds->GetPointData()->GetArray(arrayName);

    vector<int> fieldDimVec(3, 0);
    fieldDimVec[0] = fieldDim.x;
    fieldDimVec[1] = fieldDim.y;
    fieldDimVec[2] = fieldDim.z;

    vector<int> pointOrderVec = pointOrder(_plane);
    vector<int> dimOrderVec = dimOrder(_plane);

    vector<int> dim(3, 0);
    dim[0] = fieldDimVec[dimOrderVec[0]];
    dim[1] = fieldDimVec[dimOrderVec[1]];
    dim[2] = fieldDimVec[dimOrderVec[2]];

    Point3D pt;
    vector<int> ptVec(3, 0);
    Point3D ptN;
    vector<int> ptNVec(3, 0);

    int k = 0;
    int pc = 0;
    long idxPt;

    for (int i = 0; i < dim[0]; ++i)
        for (int j = 0; j < dim[1]; ++j) {
            ptVec[0] = i;
            ptVec[1] = j;
            ptVec[2] = _pos;

            pt.x = ptVec[pointOrderVec[0]];
            pt.y = ptVec[pointOrderVec[1]];
            pt.z = ptVec[pointOrderVec[2]];
            Coordinates3D<double> hexCoords = HexCoordXY(pt.x, pt.y, pt.z);
            idxPt = indexPoint3D(pt);
            if (pt.z % 3 == 0) { // z divisible by 3
                if (pt.y % 2) { //y_odd
                    if (pt.x - 1 >= 0) {
                        ptN.x = pt.x - 1;
                        ptN.y = pt.y;
                        ptN.z = pt.z;
                        if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                            Coordinates3D<double> hexCoordsP1 = hexagonVertices[4] + hexCoords;
                            Coordinates3D<double> hexCoordsP2 = hexagonVertices[5] + hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x, hexCoordsP1.y, 0.0);
                            points->InsertNextPoint(hexCoordsP2.x, hexCoordsP2.y, 0.0);
                            pc += 2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc - 2);
                            lines->InsertCellPoint(pc - 1);
                        }
                    }
                    if (pt.x - 1 >= 0 && pt.y + 1 < dim[1]) {
                        ptN.x = pt.x - 1;
                        ptN.y = pt.y + 1;
                        ptN.z = pt.z;
                        if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                            Coordinates3D<double> hexCoordsP1 = hexagonVertices[5] + hexCoords;
                            Coordinates3D<double> hexCoordsP2 = hexagonVertices[0] + hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x, hexCoordsP1.y, 0.0);
                            points->InsertNextPoint(hexCoordsP2.x, hexCoordsP2.y, 0.0);
                            pc += 2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc - 2);
                            lines->InsertCellPoint(pc - 1);
                        }
                    }
                    if (pt.y + 1 < dim[1]) {
                        ptN.x = pt.x;
                        ptN.y = pt.y + 1;
                        ptN.z = pt.z;
                        if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                            Coordinates3D<double> hexCoordsP1 = hexagonVertices[0] + hexCoords;
                            Coordinates3D<double> hexCoordsP2 = hexagonVertices[1] + hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x, hexCoordsP1.y, 0.0);
                            points->InsertNextPoint(hexCoordsP2.x, hexCoordsP2.y, 0.0);
                            pc += 2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc - 2);
                            lines->InsertCellPoint(pc - 1);
                        }
                    }
                    if (pt.x + 1 < dim[0]) {
                        ptN.x = pt.x + 1;
                        ptN.y = pt.y;
                        ptN.z = pt.z;
                        if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                            Coordinates3D<double> hexCoordsP1 = hexagonVertices[1] + hexCoords;
                            Coordinates3D<double> hexCoordsP2 = hexagonVertices[2] + hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x, hexCoordsP1.y, 0.0);
                            points->InsertNextPoint(hexCoordsP2.x, hexCoordsP2.y, 0.0);
                            pc += 2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc - 2);
                            lines->InsertCellPoint(pc - 1);
                        }
                    }
                    if (pt.y - 1 >= 0) {
                        ptN.x = pt.x;
                        ptN.y = pt.y - 1;
                        ptN.z = pt.z;
                        if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                            Coordinates3D<double> hexCoordsP1 = hexagonVertices[2] + hexCoords;
                            Coordinates3D<double> hexCoordsP2 = hexagonVertices[3] + hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x, hexCoordsP1.y, 0.0);
                            points->InsertNextPoint(hexCoordsP2.x, hexCoordsP2.y, 0.0);
                            pc += 2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc - 2);
                            lines->InsertCellPoint(pc - 1);
                        }
                    }
                    if (pt.x - 1 >= 0 && pt.y - 1 >= 0) {
                        ptN.x = pt.x - 1;
                        ptN.y = pt.y - 1;
                        ptN.z = pt.z;
                        if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                            Coordinates3D<double> hexCoordsP1 = hexagonVertices[3] + hexCoords;
                            Coordinates3D<double> hexCoordsP2 = hexagonVertices[4] + hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x, hexCoordsP1.y, 0.0);
                            points->InsertNextPoint(hexCoordsP2.x, hexCoordsP2.y, 0.0);
                            pc += 2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc - 2);
                            lines->InsertCellPoint(pc - 1);
                        }
                    }

                } else {//y_even

                    if (pt.x - 1 >= 0) {
                        ptN.x = pt.x - 1;
                        ptN.y = pt.y;
                        ptN.z = pt.z;
                        if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {
                            //if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                            Coordinates3D<double> hexCoordsP1 = hexagonVertices[4] + hexCoords;
                            Coordinates3D<double> hexCoordsP2 = hexagonVertices[5] + hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x, hexCoordsP1.y, 0.0);
                            points->InsertNextPoint(hexCoordsP2.x, hexCoordsP2.y, 0.0);
                            pc += 2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc - 2);
                            lines->InsertCellPoint(pc - 1);
                        }
                    }
                    if (pt.y + 1 < dim[1]) {
                        ptN.x = pt.x;
                        ptN.y = pt.y + 1;
                        ptN.z = pt.z;
                        if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                            Coordinates3D<double> hexCoordsP1 = hexagonVertices[5] + hexCoords;
                            Coordinates3D<double> hexCoordsP2 = hexagonVertices[0] + hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x, hexCoordsP1.y, 0.0);
                            points->InsertNextPoint(hexCoordsP2.x, hexCoordsP2.y, 0.0);
                            pc += 2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc - 2);
                            lines->InsertCellPoint(pc - 1);
                        }
                    }
                    if (pt.x + 1 < dim[0] && pt.y + 1 < dim[1]) {
                        ptN.x = pt.x + 1;
                        ptN.y = pt.y + 1;
                        ptN.z = pt.z;
                        if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                            Coordinates3D<double> hexCoordsP1 = hexagonVertices[0] + hexCoords;
                            Coordinates3D<double> hexCoordsP2 = hexagonVertices[1] + hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x, hexCoordsP1.y, 0.0);
                            points->InsertNextPoint(hexCoordsP2.x, hexCoordsP2.y, 0.0);
                            pc += 2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc - 2);
                            lines->InsertCellPoint(pc - 1);
                        }
                    }
                    if (pt.x + 1 < dim[0]) {
                        ptN.x = pt.x + 1;
                        ptN.y = pt.y;
                        ptN.z = pt.z;
                        if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                            Coordinates3D<double> hexCoordsP1 = hexagonVertices[1] + hexCoords;
                            Coordinates3D<double> hexCoordsP2 = hexagonVertices[2] + hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x, hexCoordsP1.y, 0.0);
                            points->InsertNextPoint(hexCoordsP2.x, hexCoordsP2.y, 0.0);
                            pc += 2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc - 2);
                            lines->InsertCellPoint(pc - 1);
                        }
                    }
                    if (pt.x + 1 < dim[0] && pt.y - 1 >= 0) {
                        ptN.x = pt.x + 1;
                        ptN.y = pt.y - 1;
                        ptN.z = pt.z;
                        if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                            Coordinates3D<double> hexCoordsP1 = hexagonVertices[2] + hexCoords;
                            Coordinates3D<double> hexCoordsP2 = hexagonVertices[3] + hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x, hexCoordsP1.y, 0.0);
                            points->InsertNextPoint(hexCoordsP2.x, hexCoordsP2.y, 0.0);
                            pc += 2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc - 2);
                            lines->InsertCellPoint(pc - 1);
                        }
                    }

                    if (pt.y - 1 >= 0) {
                        ptN.x = pt.x;
                        ptN.y = pt.y - 1;
                        ptN.z = pt.z;
                        if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                            Coordinates3D<double> hexCoordsP1 = hexagonVertices[3] + hexCoords;
                            Coordinates3D<double> hexCoordsP2 = hexagonVertices[4] + hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x, hexCoordsP1.y, 0.0);
                            points->InsertNextPoint(hexCoordsP2.x, hexCoordsP2.y, 0.0);
                            pc += 2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc - 2);
                            lines->InsertCellPoint(pc - 1);
                        }
                    }
                }

            } else { //apparently for  pt.z%3==1 and pt.z%3==2 xy hex shifts are the same so one code serves them both
                cerr<<"pt.z % 3 != 0"<<endl;
                if (pt.y % 2) { //y_odd
                    if (pt.x - 1 >= 0) {
                        ptN.x = pt.x - 1;
                        ptN.y = pt.y;
                        ptN.z = pt.z;
                        if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                            Coordinates3D<double> hexCoordsP1 = hexagonVertices[4] + hexCoords;
                            Coordinates3D<double> hexCoordsP2 = hexagonVertices[5] + hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x, hexCoordsP1.y, 0.0);
                            points->InsertNextPoint(hexCoordsP2.x, hexCoordsP2.y, 0.0);
                            pc += 2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc - 2);
                            lines->InsertCellPoint(pc - 1);
                        }
                    }
                    if (pt.x - 1 >= 0 && pt.y + 1 < dim[1]) {
                        ptN.x = pt.x + 1;
                        ptN.y = pt.y - 1;
                        ptN.z = pt.z;
                        if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                            Coordinates3D<double> hexCoordsP1 = hexagonVertices[2] + hexCoords;
                            Coordinates3D<double> hexCoordsP2 = hexagonVertices[3] + hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x, hexCoordsP1.y, 0.0);
                            points->InsertNextPoint(hexCoordsP2.x, hexCoordsP2.y, 0.0);
                            pc += 2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc - 2);
                            lines->InsertCellPoint(pc - 1);
                        }
                    }
                    if (pt.y + 1 < dim[1]) {
                        ptN.x = pt.x;
                        ptN.y = pt.y + 1;
                        ptN.z = pt.z;
                        if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                            Coordinates3D<double> hexCoordsP1 = hexagonVertices[5] + hexCoords;
                            Coordinates3D<double> hexCoordsP2 = hexagonVertices[0] + hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x, hexCoordsP1.y, 0.0);
                            points->InsertNextPoint(hexCoordsP2.x, hexCoordsP2.y, 0.0);
                            pc += 2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc - 2);
                            lines->InsertCellPoint(pc - 1);
                        }
                    }
                    if (pt.x + 1 < dim[0]) {
                        ptN.x = pt.x + 1;
                        ptN.y = pt.y;
                        ptN.z = pt.z;
                        if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                            Coordinates3D<double> hexCoordsP1 = hexagonVertices[1] + hexCoords;
                            Coordinates3D<double> hexCoordsP2 = hexagonVertices[2] + hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x, hexCoordsP1.y, 0.0);
                            points->InsertNextPoint(hexCoordsP2.x, hexCoordsP2.y, 0.0);
                            pc += 2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc - 2);
                            lines->InsertCellPoint(pc - 1);
                        }
                    }
                    if (pt.y - 1 >= 0) {
                        ptN.x = pt.x;
                        ptN.y = pt.y - 1;
                        ptN.z = pt.z;
                        if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                            Coordinates3D<double> hexCoordsP1 = hexagonVertices[3] + hexCoords;
                            Coordinates3D<double> hexCoordsP2 = hexagonVertices[4] + hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x, hexCoordsP1.y, 0.0);
                            points->InsertNextPoint(hexCoordsP2.x, hexCoordsP2.y, 0.0);
                            pc += 2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc - 2);
                            lines->InsertCellPoint(pc - 1);
                        }
                    }
                    if (pt.x - 1 >= 0 && pt.y - 1 >= 0) {
                        ptN.x = pt.x + 1;
                        ptN.y = pt.y + 1;
                        ptN.z = pt.z;
                        if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                            Coordinates3D<double> hexCoordsP1 = hexagonVertices[0] + hexCoords;
                            Coordinates3D<double> hexCoordsP2 = hexagonVertices[1] + hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x, hexCoordsP1.y, 0.0);
                            points->InsertNextPoint(hexCoordsP2.x, hexCoordsP2.y, 0.0);
                            pc += 2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc - 2);
                            lines->InsertCellPoint(pc - 1);
                        }
                    }

                } else {//y_even

                    if (pt.x - 1 >= 0) {
                        ptN.x = pt.x - 1;
                        ptN.y = pt.y;
                        ptN.z = pt.z;
                        if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {
                            //if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                            Coordinates3D<double> hexCoordsP1 = hexagonVertices[4] + hexCoords;
                            Coordinates3D<double> hexCoordsP2 = hexagonVertices[5] + hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x, hexCoordsP1.y, 0.0);
                            points->InsertNextPoint(hexCoordsP2.x, hexCoordsP2.y, 0.0);
                            pc += 2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc - 2);
                            lines->InsertCellPoint(pc - 1);
                        }
                    }
                    if (pt.x - 1 >= 0 && pt.y + 1 < dim[1]) {
                        ptN.x = pt.x - 1;
                        ptN.y = pt.y + 1;
                        ptN.z = pt.z;
                        if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                            Coordinates3D<double> hexCoordsP1 = hexagonVertices[5] + hexCoords;
                            Coordinates3D<double> hexCoordsP2 = hexagonVertices[0] + hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x, hexCoordsP1.y, 0.0);
                            points->InsertNextPoint(hexCoordsP2.x, hexCoordsP2.y, 0.0);
                            pc += 2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc - 2);
                            lines->InsertCellPoint(pc - 1);
                        }
                    }
                    if (pt.y + 1 < dim[1]) {
                        ptN.x = pt.x ;
                        ptN.y = pt.y + 1;
                        ptN.z = pt.z;
                        if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                            Coordinates3D<double> hexCoordsP1 = hexagonVertices[0] + hexCoords;
                            Coordinates3D<double> hexCoordsP2 = hexagonVertices[1] + hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x, hexCoordsP1.y, 0.0);
                            points->InsertNextPoint(hexCoordsP2.x, hexCoordsP2.y, 0.0);
                            pc += 2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc - 2);
                            lines->InsertCellPoint(pc - 1);
                        }
                    }
                    if (pt.x + 1 < dim[0]) {
                        ptN.x = pt.x + 1;
                        ptN.y = pt.y;
                        ptN.z = pt.z;
                        if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                            Coordinates3D<double> hexCoordsP1 = hexagonVertices[1] + hexCoords;
                            Coordinates3D<double> hexCoordsP2 = hexagonVertices[2] + hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x, hexCoordsP1.y, 0.0);
                            points->InsertNextPoint(hexCoordsP2.x, hexCoordsP2.y, 0.0);
                            pc += 2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc - 2);
                            lines->InsertCellPoint(pc - 1);
                        }
                    }
                    if (pt.y - 1 >= 0) {
                        ptN.x = pt.x;
                        ptN.y = pt.y - 1;
                        ptN.z = pt.z;
                        if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                            Coordinates3D<double> hexCoordsP1 = hexagonVertices[2] + hexCoords;
                            Coordinates3D<double> hexCoordsP2 = hexagonVertices[3] + hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x, hexCoordsP1.y, 0.0);
                            points->InsertNextPoint(hexCoordsP2.x, hexCoordsP2.y, 0.0);
                            pc += 2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc - 2);
                            lines->InsertCellPoint(pc - 1);
                        }
                    }

                    if (pt.x - 1 >= 0 && pt.y - 1 >= 0) {
                        ptN.x = pt.x -1 ;
                        ptN.y = pt.y - 1;
                        ptN.z = pt.z;
                        if (idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN))) {

                            Coordinates3D<double> hexCoordsP1 = hexagonVertices[3] + hexCoords;
                            Coordinates3D<double> hexCoordsP2 = hexagonVertices[4] + hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x, hexCoordsP1.y, 0.0);
                            points->InsertNextPoint(hexCoordsP2.x, hexCoordsP2.y, 0.0);
                            pc += 2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc - 2);
                            lines->InsertCellPoint(pc - 1);
                        }
                    }
                }
            }
        }
}

void FieldExtractorCML::fillBorderData2DHex(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr,
                                            std::string _plane, int _pos) {
    fillBorder2DHex("CellId", _pointArrayAddr, _linesArrayAddr, _plane, _pos);
}

void
FieldExtractorCML::fillClusterBorderData2DHex(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr,
                                              std::string _plane, int _pos) {
    fillBorder2DHex("ClusterId", _pointArrayAddr, _linesArrayAddr, _plane, _pos);
}

bool FieldExtractorCML::fillConFieldData2DHex(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _hexCellsArrayAddr,
                                              vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName,
                                              std::string _plane, int _pos) {

    vtkDoubleArray *conArrayRead = (vtkDoubleArray *) lds->GetPointData()->GetArray(_conFieldName.c_str());

    if (!conArrayRead)
        return false;

    vtkDoubleArray *conArray = (vtkDoubleArray *) _conArrayAddr;

    vtkCellArray *_hexCellsArray = (vtkCellArray *) _hexCellsArrayAddr;

    vtkPoints *_pointsArray = (vtkPoints *) _pointsArrayAddr;

    vector<int> fieldDimVec(3, 0);
    fieldDimVec[0] = fieldDim.x;
    fieldDimVec[1] = fieldDim.y;
    fieldDimVec[2] = fieldDim.z;

    vector<int> pointOrderVec = pointOrder(_plane);
    vector<int> dimOrderVec = dimOrder(_plane);

    vector<int> dim(3, 0);
    dim[0] = fieldDimVec[dimOrderVec[0]];
    dim[1] = fieldDimVec[dimOrderVec[1]];
    dim[2] = fieldDimVec[dimOrderVec[2]];

    int offset = 0;

    Point3D pt;
    vector<int> ptVec(3, 0);

    double con;
    long pc = 0;
    //when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
    for (int j = 0; j < dim[1]; ++j)
        for (int i = 0; i < dim[0]; ++i) {
            ptVec[0] = i;
            ptVec[1] = j;
            ptVec[2] = _pos;

            pt.x = ptVec[pointOrderVec[0]];
            pt.y = ptVec[pointOrderVec[1]];
            pt.z = ptVec[pointOrderVec[2]];

            if (i == dim[0] || j == dim[1]) {
                con = 0.0;
            } else {
                con = conArrayRead->GetValue(indexPoint3D(pt));
            }
            Coordinates3D<double> hexCoords = HexCoordXY(pt.x, pt.y, pt.z);
            for (int idx = 0; idx < 6; ++idx) {

                Coordinates3D<double> hexagonVertex = hexagonVertices[idx] + hexCoords;

                _pointsArray->InsertNextPoint(hexagonVertex.x, hexagonVertex.y, 0.0);
            }
            pc += 6;
            vtkIdType cellId = _hexCellsArray->InsertNextCell(6);
            _hexCellsArray->InsertCellPoint(pc - 6);
            _hexCellsArray->InsertCellPoint(pc - 5);
            _hexCellsArray->InsertCellPoint(pc - 4);
            _hexCellsArray->InsertCellPoint(pc - 3);
            _hexCellsArray->InsertCellPoint(pc - 2);
            _hexCellsArray->InsertCellPoint(pc - 1);

            conArray->InsertNextValue(con);
        }
    return true;
}

bool
FieldExtractorCML::fillScalarFieldData2DHex(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _hexCellsArrayAddr,
                                            vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName,
                                            std::string _plane, int _pos) {

    return fillConFieldData2DHex(_conArrayAddr, _hexCellsArrayAddr, _pointsArrayAddr, _conFieldName, _plane, _pos);
}

bool FieldExtractorCML::fillScalarFieldCellLevelData2DHex(vtk_obj_addr_int_t _conArrayAddr,
                                                          vtk_obj_addr_int_t _hexCellsArrayAddr,
                                                          vtk_obj_addr_int_t _pointsArrayAddr,
                                                          std::string _conFieldName, std::string _plane, int _pos) {

    return fillConFieldData2DHex(_conArrayAddr, _hexCellsArrayAddr, _pointsArrayAddr, _conFieldName, _plane, _pos);
}

bool
FieldExtractorCML::fillConFieldData2D(vtk_obj_addr_int_t _conArrayAddr, std::string _conFieldName, std::string _plane,
                                      int _pos) {
    vtkDoubleArray *conArrayRead = (vtkDoubleArray *) lds->GetPointData()->GetArray(_conFieldName.c_str());

    if (!conArrayRead)
        return false;

    vtkDoubleArray *conArray = (vtkDoubleArray *) _conArrayAddr;

    vector<int> fieldDimVec(3, 0);
    fieldDimVec[0] = fieldDim.x;
    fieldDimVec[1] = fieldDim.y;
    fieldDimVec[2] = fieldDim.z;

    vector<int> pointOrderVec = pointOrder(_plane);
    vector<int> dimOrderVec = dimOrder(_plane);

    vector<int> dim(3, 0);
    dim[0] = fieldDimVec[dimOrderVec[0]];
    dim[1] = fieldDimVec[dimOrderVec[1]];
    dim[2] = fieldDimVec[dimOrderVec[2]];

    conArray->SetNumberOfValues((dim[1] + 2) * (dim[0] + 1));
    //For some reasons the points x=0 are eaten up (don't know why).
    //So we just populate concentration 0.0.
    int offset = 0;
    for (int i = 0; i < dim[0] + 1; ++i) {
        conArray->SetValue(offset, 0.0);
        ++offset;
    }

    Point3D pt;
    vector<int> ptVec(3, 0);

    double con;
    //when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
    for (int j = 0; j < dim[1] + 1; ++j)
        for (int i = 0; i < dim[0] + 1; ++i) {
            ptVec[0] = i;
            ptVec[1] = j;
            ptVec[2] = _pos;

            pt.x = ptVec[pointOrderVec[0]];
            pt.y = ptVec[pointOrderVec[1]];
            pt.z = ptVec[pointOrderVec[2]];

            if (i == dim[0] || j == dim[1]) {
                con = 0.0;
            } else {
                con = conArrayRead->GetValue(indexPoint3D(pt));
            }
            conArray->SetValue(offset, con);
            ++offset;
        }
    return true;
}

bool FieldExtractorCML::fillScalarFieldData2D(vtk_obj_addr_int_t _conArrayAddr, std::string _conFieldName,
                                              std::string _plane, int _pos) {

    return fillConFieldData2D(_conArrayAddr, _conFieldName, _plane, _pos);
}

bool FieldExtractorCML::fillScalarFieldData2DCartesian(vtk_obj_addr_int_t _conArrayAddr,
                                                       vtk_obj_addr_int_t _cartesianCellsArrayAddr,
                                                       vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName,
                                                       std::string _plane, int _pos) {
    return fillConFieldData2DCartesian(_conArrayAddr, _cartesianCellsArrayAddr, _pointsArrayAddr, _conFieldName, _plane,
                                       _pos);
}

bool FieldExtractorCML::fillConFieldData2DCartesian(vtk_obj_addr_int_t _conArrayAddr,
                                                    vtk_obj_addr_int_t _cartesianCellsArrayAddr,
                                                    vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName,
                                                    std::string _plane, int _pos) {
    vtkDoubleArray *conArrayRead = (vtkDoubleArray *) lds->GetPointData()->GetArray(_conFieldName.c_str());

    if (!conArrayRead)
        return false;

    vtkDoubleArray *conArray = (vtkDoubleArray *) _conArrayAddr;

    vtkCellArray *_cartesianCellsArray = (vtkCellArray *) _cartesianCellsArrayAddr;

    vtkPoints *_pointsArray = (vtkPoints *) _pointsArrayAddr;

    vector<int> fieldDimVec(3, 0);
    fieldDimVec[0] = fieldDim.x;
    fieldDimVec[1] = fieldDim.y;
    fieldDimVec[2] = fieldDim.z;

    vector<int> pointOrderVec = pointOrder(_plane);
    vector<int> dimOrderVec = dimOrder(_plane);

    vector<int> dim(3, 0);
    dim[0] = fieldDimVec[dimOrderVec[0]];
    dim[1] = fieldDimVec[dimOrderVec[1]];
    dim[2] = fieldDimVec[dimOrderVec[2]];


    int offset = 0;

    Point3D pt;
    vector<int> ptVec(3, 0);

    double con;
    long pc = 0;
    //when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned


    for (int j = 0; j < dim[1]; ++j)
        for (int i = 0; i < dim[0]; ++i) {
            ptVec[0] = i;
            ptVec[1] = j;
            ptVec[2] = _pos;

            pt.x = ptVec[pointOrderVec[0]];
            pt.y = ptVec[pointOrderVec[1]];
            pt.z = ptVec[pointOrderVec[2]];

            if (i == dim[0] || j == dim[1]) {
                con = 0.0;
            } else {
                con = conArrayRead->GetValue(indexPoint3D(pt));
            }

            Coordinates3D<double> coords(ptVec[0], ptVec[1],
                                         0); // notice that we are drawing pixels from other planes on a xy plan so we use ptVec instead of pt. pt is absolute position of the point ptVec is for projection purposes

            for (int idx = 0; idx < 4; ++idx) {
                Coordinates3D<double> cartesianVertex = cartesianVertices[idx] + coords;
                _pointsArray->InsertNextPoint(cartesianVertex.x, cartesianVertex.y, 0.0);
            }

            pc += 4;
            vtkIdType cellId = _cartesianCellsArray->InsertNextCell(4);
            _cartesianCellsArray->InsertCellPoint(pc - 4);
            _cartesianCellsArray->InsertCellPoint(pc - 3);
            _cartesianCellsArray->InsertCellPoint(pc - 2);
            _cartesianCellsArray->InsertCellPoint(pc - 1);

            conArray->InsertNextValue(con);
            ++offset;
        }

    return true;

}


bool FieldExtractorCML::fillScalarFieldCellLevelData2D(vtk_obj_addr_int_t _conArrayAddr, std::string _conFieldName,
                                                       std::string _plane, int _pos) {

    return fillConFieldData2D(_conArrayAddr, _conFieldName, _plane, _pos);
}

bool FieldExtractorCML::fillScalarFieldCellLevelData2DCartesian(vtk_obj_addr_int_t _conArrayAddr,
                                                                vtk_obj_addr_int_t _cartesianCellsArrayAddr,
                                                                vtk_obj_addr_int_t _pointsArrayAddr,
                                                                std::string _conFieldName, std::string _plane,
                                                                int _pos) {
    return fillConFieldData2DCartesian(_conArrayAddr, _cartesianCellsArrayAddr, _pointsArrayAddr, _conFieldName, _plane,
                                       _pos);
}


bool
FieldExtractorCML::fillVectorFieldData2D(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr,
                                         std::string _fieldName, std::string _plane, int _pos) {
    vtkFloatArray *vectorArray = (vtkFloatArray *) _vectorArrayIntAddr;
    vtkPoints *pointsArray = (vtkPoints *) _pointsArrayIntAddr;

    vtkFloatArray *vecArrayRead = (vtkFloatArray *) lds->GetPointData()->GetArray(_fieldName.c_str());

    if (!vecArrayRead)
        return false;

    vector<int> fieldDimVec(3, 0);
    fieldDimVec[0] = fieldDim.x;
    fieldDimVec[1] = fieldDim.y;
    fieldDimVec[2] = fieldDim.z;

    vector<int> pointOrderVec = pointOrder(_plane);
    vector<int> dimOrderVec = dimOrder(_plane);

    vector<int> dim(3, 0);
    dim[0] = fieldDimVec[dimOrderVec[0]];
    dim[1] = fieldDimVec[dimOrderVec[1]];
    dim[2] = fieldDimVec[dimOrderVec[2]];

    Point3D pt;
    vector<int> ptVec(3, 0);

    float vecTmp[3];
    double vecTmpCoord[3];
    //double con;

    int offset = 0;

    for (int j = 0; j < dim[1]; ++j)
        for (int i = 0; i < dim[0]; ++i) {
            ptVec[0] = i;
            ptVec[1] = j;
            ptVec[2] = _pos;

            pt.x = ptVec[pointOrderVec[0]];
            pt.y = ptVec[pointOrderVec[1]];
            pt.z = ptVec[pointOrderVec[2]];

            vecArrayRead->GetTuple(indexPoint3D(pt), vecTmpCoord);

            if (vecTmpCoord[0] != 0.0 || vecTmpCoord[1] != 0.0 || vecTmpCoord[2] != 0.0) {
                pointsArray->InsertPoint(offset, ptVec[0], ptVec[1], 0);
                vectorArray->InsertTuple3(offset, vecTmpCoord[pointOrderVec[0]], vecTmpCoord[pointOrderVec[1]], 0);
                ++offset;
            }
        }
    return true;
}

bool FieldExtractorCML::fillVectorFieldData2DHex(vtk_obj_addr_int_t _pointsArrayIntAddr,
                                                 vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName,
                                                 std::string _plane, int _pos) {
    vtkFloatArray *vectorArray = (vtkFloatArray *) _vectorArrayIntAddr;
    vtkPoints *pointsArray = (vtkPoints *) _pointsArrayIntAddr;

    vtkFloatArray *vecArrayRead = (vtkFloatArray *) lds->GetPointData()->GetArray(_fieldName.c_str());

    if (!vecArrayRead)
        return false;

    vector<int> fieldDimVec(3, 0);
    fieldDimVec[0] = fieldDim.x;
    fieldDimVec[1] = fieldDim.y;
    fieldDimVec[2] = fieldDim.z;

    vector<int> pointOrderVec = pointOrder(_plane);
    vector<int> dimOrderVec = dimOrder(_plane);

    vector<int> dim(3, 0);
    dim[0] = fieldDimVec[dimOrderVec[0]];
    dim[1] = fieldDimVec[dimOrderVec[1]];
    dim[2] = fieldDimVec[dimOrderVec[2]];

    Point3D pt;
    vector<int> ptVec(3, 0);

    float vecTmp[3];
    double vecTmpCoord[3];

    int offset = 0;

    for (int j = 0; j < dim[1]; ++j)
        for (int i = 0; i < dim[0]; ++i) {
            ptVec[0] = i;
            ptVec[1] = j;
            ptVec[2] = _pos;

            pt.x = ptVec[pointOrderVec[0]];
            pt.y = ptVec[pointOrderVec[1]];
            pt.z = ptVec[pointOrderVec[2]];

            vecArrayRead->GetTuple(indexPoint3D(pt), vecTmpCoord);

            if (vecTmpCoord[0] != 0.0 || vecTmpCoord[1] != 0.0 || vecTmpCoord[2] != 0.0) {

                Coordinates3D<double> hexCoords = HexCoordXY(pt.x, pt.y, pt.z);
                pointsArray->InsertPoint(offset, hexCoords.x, hexCoords.y, 0.0);

                vectorArray->InsertTuple3(offset, vecTmpCoord[pointOrderVec[0]], vecTmpCoord[pointOrderVec[1]], 0);
                ++offset;
            }
        }
    return true;
}

bool
FieldExtractorCML::fillVectorFieldData3D(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr,
                                         std::string _fieldName) {

    vtkFloatArray *vectorArray = (vtkFloatArray *) _vectorArrayIntAddr;
    vtkPoints *pointsArray = (vtkPoints *) _pointsArrayIntAddr;

    vtkFloatArray *vecArrayRead = (vtkFloatArray *) lds->GetPointData()->GetArray(_fieldName.c_str());

    if (!vecArrayRead)
        return false;

    Point3D pt;
    vector<int> ptVec(3, 0);

    double vecTmp[3];

    int offset = 0;
    for (pt.z = 0; pt.z < fieldDim.z; ++pt.z)
        for (pt.y = 0; pt.y < fieldDim.y; ++pt.y)
            for (pt.x = 0; pt.x < fieldDim.x; ++pt.x) {

                vecArrayRead->GetTuple(indexPoint3D(pt), vecTmp);
                if (vecTmp[0] != 0.0 || vecTmp[1] != 0.0 || vecTmp[2] != 0.0) {
                    pointsArray->InsertPoint(offset, pt.x, pt.y, pt.z);
                    vectorArray->InsertTuple3(offset, vecTmp[0], vecTmp[1], vecTmp[2]);
                    ++offset;
                }
            }
    return true;
}


bool FieldExtractorCML::fillVectorFieldData3DHex(vtk_obj_addr_int_t _pointsArrayIntAddr,
                                                 vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName) {

    vtkFloatArray *vectorArray = (vtkFloatArray *) _vectorArrayIntAddr;
    vtkPoints *pointsArray = (vtkPoints *) _pointsArrayIntAddr;

    vtkFloatArray *vecArrayRead = (vtkFloatArray *) lds->GetPointData()->GetArray(_fieldName.c_str());

    if (!vecArrayRead)
        return false;

    Point3D pt;
    vector<int> ptVec(3, 0);

    double vecTmp[3];

    int offset = 0;
    for (pt.z = 0; pt.z < fieldDim.z; ++pt.z)
        for (pt.y = 0; pt.y < fieldDim.y; ++pt.y)
            for (pt.x = 0; pt.x < fieldDim.x; ++pt.x) {

                vecArrayRead->GetTuple(indexPoint3D(pt), vecTmp);
                if (vecTmp[0] != 0.0 || vecTmp[1] != 0.0 || vecTmp[2] != 0.0) {
                    Coordinates3D<double> hexCoords = HexCoordXY(pt.x, pt.y, pt.z);
                    pointsArray->InsertPoint(offset, hexCoords.x, hexCoords.y, hexCoords.z);

                    vectorArray->InsertTuple3(offset, vecTmp[0], vecTmp[1], vecTmp[2]);
                    ++offset;
                }
            }
    return true;
}


bool FieldExtractorCML::fillVectorFieldCellLevelData2D(vtk_obj_addr_int_t _pointsArrayIntAddr,
                                                       vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName,
                                                       std::string _plane, int _pos) {
    vtkFloatArray *vectorArray = (vtkFloatArray *) _vectorArrayIntAddr;
    vtkPoints *pointsArray = (vtkPoints *) _pointsArrayIntAddr;

    set<long> visitedCells;

    vtkLongArray *idArray = (vtkLongArray *) lds->GetPointData()->GetArray("CellId");

    vtkFloatArray *vecArrayRead = (vtkFloatArray *) lds->GetPointData()->GetArray(_fieldName.c_str());

    if (!vecArrayRead)
        return false;

    vector<int> fieldDimVec(3, 0);
    fieldDimVec[0] = fieldDim.x;
    fieldDimVec[1] = fieldDim.y;
    fieldDimVec[2] = fieldDim.z;

    vector<int> pointOrderVec = pointOrder(_plane);
    vector<int> dimOrderVec = dimOrder(_plane);

    vector<int> dim(3, 0);
    dim[0] = fieldDimVec[dimOrderVec[0]];
    dim[1] = fieldDimVec[dimOrderVec[1]];
    dim[2] = fieldDimVec[dimOrderVec[2]];

    Point3D pt;
    vector<int> ptVec(3, 0);

    long cellId;
    long idx;
    Coordinates3D<float> vecTmp;
    double vecTmpCoord[3];

    int offset = 0;

    for (int j = 0; j < dim[1]; ++j)
        for (int i = 0; i < dim[0]; ++i) {
            ptVec[0] = i;
            ptVec[1] = j;
            ptVec[2] = _pos;

            pt.x = ptVec[pointOrderVec[0]];
            pt.y = ptVec[pointOrderVec[1]];
            pt.z = ptVec[pointOrderVec[2]];

            idx = indexPoint3D(pt);

            cellId = idArray->GetValue(idx);

            if (cellId) {
                //check if this cell is in the set of visited Cells
                if (visitedCells.find(cellId) != visitedCells.end()) {
                    continue; //cell have been visited
                } else {
                    //this is first time we visit given cell

                    vecArrayRead->GetTuple(idx, vecTmpCoord);

                    if (vecTmpCoord[0] != 0.0 || vecTmpCoord[1] != 0.0 || vecTmpCoord[2] != 0.0) {


                        pointsArray->InsertPoint(offset, ptVec[0], ptVec[1], 0);
                        vectorArray->InsertTuple3(offset, vecTmpCoord[pointOrderVec[0]], vecTmpCoord[pointOrderVec[1]],
                                                  0);
                        ++offset;
                    }
                    visitedCells.insert(cellId);
                }
            }
        }
    return true;
}

bool FieldExtractorCML::fillVectorFieldCellLevelData2DHex(vtk_obj_addr_int_t _pointsArrayIntAddr,
                                                          vtk_obj_addr_int_t _vectorArrayIntAddr,
                                                          std::string _fieldName, std::string _plane, int _pos) {

    vtkFloatArray *vectorArray = (vtkFloatArray *) _vectorArrayIntAddr;
    vtkPoints *pointsArray = (vtkPoints *) _pointsArrayIntAddr;

    set<long> visitedCells;

    vtkLongArray *idArray = (vtkLongArray *) lds->GetPointData()->GetArray("CellId");

    vtkFloatArray *vecArrayRead = (vtkFloatArray *) lds->GetPointData()->GetArray(_fieldName.c_str());

    if (!vecArrayRead)
        return false;

    vector<int> fieldDimVec(3, 0);
    fieldDimVec[0] = fieldDim.x;
    fieldDimVec[1] = fieldDim.y;
    fieldDimVec[2] = fieldDim.z;

    vector<int> pointOrderVec = pointOrder(_plane);
    vector<int> dimOrderVec = dimOrder(_plane);

    vector<int> dim(3, 0);
    dim[0] = fieldDimVec[dimOrderVec[0]];
    dim[1] = fieldDimVec[dimOrderVec[1]];
    dim[2] = fieldDimVec[dimOrderVec[2]];

    Point3D pt;
    vector<int> ptVec(3, 0);
    long cellId;
    long idx;
    Coordinates3D<float> vecTmp;
    double vecTmpCoord[3];

    int offset = 0;

    for (int j = 0; j < dim[1]; ++j)
        for (int i = 0; i < dim[0]; ++i) {
            ptVec[0] = i;
            ptVec[1] = j;
            ptVec[2] = _pos;

            pt.x = ptVec[pointOrderVec[0]];
            pt.y = ptVec[pointOrderVec[1]];
            pt.z = ptVec[pointOrderVec[2]];

            idx = indexPoint3D(pt);

            cellId = idArray->GetValue(idx);

            if (cellId) {
                //check if this cell is in the set of visited Cells
                if (visitedCells.find(cellId) != visitedCells.end()) {
                    continue; //cell have been visited
                } else {
                    //this is first time we visit given cell

                    vecArrayRead->GetTuple(idx, vecTmpCoord);

                    if (vecTmpCoord[0] != 0.0 || vecTmpCoord[1] != 0.0 || vecTmpCoord[2] != 0.0) {

                        Coordinates3D<double> hexCoords = HexCoordXY(pt.x, pt.y, pt.z);
                        pointsArray->InsertPoint(offset, hexCoords.x, hexCoords.y, 0.0);

                        vectorArray->InsertTuple3(offset, vecTmpCoord[pointOrderVec[0]], vecTmpCoord[pointOrderVec[1]],
                                                  0);
                        ++offset;
                    }
                    visitedCells.insert(cellId);
                }
            }
        }
    return true;
}

bool FieldExtractorCML::fillVectorFieldCellLevelData3D(vtk_obj_addr_int_t _pointsArrayIntAddr,
                                                       vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName) {
    vtkFloatArray *vectorArray = (vtkFloatArray *) _vectorArrayIntAddr;
    vtkPoints *pointsArray = (vtkPoints *) _pointsArrayIntAddr;

    set<long> visitedCells;

    vtkLongArray *idArray = (vtkLongArray *) lds->GetPointData()->GetArray("CellId");

    vtkFloatArray *vecArrayRead = (vtkFloatArray *) lds->GetPointData()->GetArray(_fieldName.c_str());

    if (!vecArrayRead)
        return false;

    Point3D pt;
    vector<int> ptVec(3, 0);
    long cellId;
    long idx;

    double vecTmp[3];

    int offset = 0;
    for (pt.z = 0; pt.z < fieldDim.z; ++pt.z)
        for (pt.y = 0; pt.y < fieldDim.y; ++pt.y)
            for (pt.x = 0; pt.x < fieldDim.x; ++pt.x) {

                idx = indexPoint3D(pt);

                cellId = idArray->GetValue(idx);

                if (cellId) {
                    //check if this cell is in the set of visited Cells
                    if (visitedCells.find(cellId) != visitedCells.end()) {
                        continue; //cell have been visited
                    } else {
                        //this is first time we visit given cell
                        vecArrayRead->GetTuple(idx, vecTmp);
                        if (vecTmp[0] != 0.0 || vecTmp[1] != 0.0 || vecTmp[2] != 0.0) {


                            pointsArray->InsertPoint(offset, pt.x, pt.y, pt.z);
                            vectorArray->InsertTuple3(offset, vecTmp[0], vecTmp[1], vecTmp[2]);
                            ++offset;
                        }
                        visitedCells.insert(cellId);
                    }
                }
            }
    return true;
}

bool FieldExtractorCML::fillVectorFieldCellLevelData3DHex(vtk_obj_addr_int_t _pointsArrayIntAddr,
                                                          vtk_obj_addr_int_t _vectorArrayIntAddr,
                                                          std::string _fieldName) {
    vtkFloatArray *vectorArray = (vtkFloatArray *) _vectorArrayIntAddr;
    vtkPoints *pointsArray = (vtkPoints *) _pointsArrayIntAddr;

    set<long> visitedCells;

    vtkLongArray *idArray = (vtkLongArray *) lds->GetPointData()->GetArray("CellId");

    vtkFloatArray *vecArrayRead = (vtkFloatArray *) lds->GetPointData()->GetArray(_fieldName.c_str());

    if (!vecArrayRead)
        return false;

    Point3D pt;
    vector<int> ptVec(3, 0);
    long cellId;
    long idx;

    double vecTmp[3];

    int offset = 0;
    for (pt.z = 0; pt.z < fieldDim.z; ++pt.z)
        for (pt.y = 0; pt.y < fieldDim.y; ++pt.y)
            for (pt.x = 0; pt.x < fieldDim.x; ++pt.x) {

                idx = indexPoint3D(pt);

                cellId = idArray->GetValue(idx);

                if (cellId) {
                    //check if this cell is in the set of visited Cells
                    if (visitedCells.find(cellId) != visitedCells.end()) {
                        continue; //cell have been visited 
                    } else {
                        //this is first time we visit given cell
                        vecArrayRead->GetTuple(idx, vecTmp);
                        if (vecTmp[0] != 0.0 || vecTmp[1] != 0.0 || vecTmp[2] != 0.0) {

                            Coordinates3D<double> hexCoords = HexCoordXY(pt.x, pt.y, pt.z);
                            pointsArray->InsertPoint(offset, hexCoords.x, hexCoords.y, hexCoords.z);
                            vectorArray->InsertTuple3(offset, vecTmp[0], vecTmp[1], vecTmp[2]);
                            ++offset;
                        }
                        visitedCells.insert(cellId);
                    }
                }
            }
    return true;
}


//vector<int> FieldExtractorCML::fillCellFieldData3D(long _cellTypeArrayAddr){
vector<int>
FieldExtractorCML::fillCellFieldData3D(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _cellIdArrayAddr,
                                       bool extractOuterShellOnly) {
    set<int> usedCellTypes;

    vtkIntArray *cellTypeArray = (vtkIntArray *) _cellTypeArrayAddr;
    vtkLongArray *cellIdArray = (vtkLongArray *) _cellIdArrayAddr;

    vtkCharArray *typeArrayRead = (vtkCharArray *) lds->GetPointData()->GetArray("CellType");
    vtkLongArray *idArrayRead = (vtkLongArray *) lds->GetPointData()->GetArray("CellId");

    cellTypeArray->SetNumberOfValues((fieldDim.x + 2) * (fieldDim.y + 2) * (fieldDim.z + 2));
    cellIdArray->SetNumberOfValues((fieldDim.x + 2) * (fieldDim.y + 2) * (fieldDim.z + 2));

    Point3D pt;
    int type;
    int id;
    long idxPt;
    int offset = 0;
    //when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
    for (int k = 0; k < fieldDim.z + 2; ++k)
        for (int j = 0; j < fieldDim.y + 2; ++j)
            for (int i = 0; i < fieldDim.x + 2; ++i) {
                if (i == 0 || i == fieldDim.x + 1 || j == 0 || j == fieldDim.y + 1 || k == 0 || k == fieldDim.z + 1) {
                    cellTypeArray->InsertValue(offset, 0);
                    cellIdArray->InsertValue(offset, 0);
                    ++offset;
                } else {
                    pt.x = i - 1;
                    pt.y = j - 1;
                    pt.z = k - 1;
                    idxPt = indexPoint3D(pt);
                    type = typeArrayRead->GetValue(idxPt);
                    id = idArrayRead->GetValue(idxPt);

                    if (type != 0)
                        usedCellTypes.insert(type);

                    cellTypeArray->InsertValue(offset, type);
                    cellIdArray->InsertValue(offset, id);

                    ++offset;
                }
            }
    return vector<int>(usedCellTypes.begin(), usedCellTypes.end());
}

bool FieldExtractorCML::fillConFieldData3D(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cellTypeArrayAddr,
                                           std::string _conFieldName, std::vector<int> *_typesInvisibeVec,
                                           bool type_indicator_only
) {
    vtkDoubleArray *conArray = (vtkDoubleArray *) _conArrayAddr;
    vtkIntArray *cellTypeArray = (vtkIntArray *) _cellTypeArrayAddr;

    vtkCharArray *typeArrayRead = (vtkCharArray *) lds->GetPointData()->GetArray("CellType");

    vtkDoubleArray *conArrayRead = (vtkDoubleArray *) lds->GetPointData()->GetArray(_conFieldName.c_str());

    if (!conArrayRead)
        return false;

    type_fcn_ptr = &FieldExtractorCML::type_value;
    if (type_indicator_only) {
        type_fcn_ptr = &FieldExtractorCML::type_indicator;
    }

    conArray->SetNumberOfValues((fieldDim.x + 2) * (fieldDim.y + 2) * (fieldDim.z + 2));
    cellTypeArray->SetNumberOfValues((fieldDim.x + 2) * (fieldDim.y + 2) * (fieldDim.z + 2));

    set<int> invisibleTypeSet(_typesInvisibeVec->begin(), _typesInvisibeVec->end());

    Point3D pt;
    long idxPt;
    double con;
    int type;
    int offset = 0;
    //when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
    for (int k = 0; k < fieldDim.z + 2; ++k)
        for (int j = 0; j < fieldDim.y + 2; ++j)
            for (int i = 0; i < fieldDim.x + 2; ++i) {
                if (i == 0 || i == fieldDim.x + 1 || j == 0 || j == fieldDim.y + 1 || k == 0 || k == fieldDim.z + 1) {
                    conArray->InsertValue(offset, 0.0);
                    cellTypeArray->InsertValue(offset, 0);
                    ++offset;
                } else {
                    pt.x = i - 1;
                    pt.y = j - 1;
                    pt.z = k - 1;

                    idxPt = indexPoint3D(pt);
                    con = conArrayRead->GetValue(idxPt);
                    type = typeArrayRead->GetValue(idxPt);
                    // applying either type indicator or putting actual type+id depending on the value of
                    // type_indicator_only flag
                    type = (this->*type_fcn_ptr)(type);

                    if (type && invisibleTypeSet.find(type) != invisibleTypeSet.end()) {
                        type = 0;
                    }

                    conArray->InsertValue(offset, con);
                    cellTypeArray->InsertValue(offset, type);
                    ++offset;
                }
            }
    return true;
}

bool FieldExtractorCML::fillScalarFieldData3D(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cellTypeArrayAddr,
                                              std::string _conFieldName, std::vector<int> *_typesInvisibeVec,
                                              bool type_indicator_only) {

    return fillConFieldData3D(_conArrayAddr, _cellTypeArrayAddr, _conFieldName, _typesInvisibeVec, type_indicator_only);
}

bool FieldExtractorCML::fillScalarFieldCellLevelData3D(vtk_obj_addr_int_t _conArrayAddr,
                                                       vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _conFieldName,
                                                       std::vector<int> *_typesInvisibeVec,
                                                       bool type_indicator_only) {

    return fillConFieldData3D(_conArrayAddr, _cellTypeArrayAddr, _conFieldName, _typesInvisibeVec, type_indicator_only);
}


bool FieldExtractorCML::readVtkStructuredPointsData(vtk_obj_addr_int_t _structuredPointsReaderAddr) {
    vtkStructuredPointsReader *reader = (vtkStructuredPointsReader *) _structuredPointsReaderAddr;
    reader->Update();

    return true;
}

void FieldExtractorCML::fillCellFieldGlyphs2D(
                vtk_obj_addr_int_t centroids_array_addr,
                vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                vtk_obj_addr_int_t cell_type_array_addr,
                std::string plane, int pos){


    // cell_id to cell type map
    unordered_map<long, int> cell_id_to_cell_type;

    unordered_map<long, list<int> > cell_id_to_coords_0;
    unordered_map<long, list<int> > cell_id_to_coords_1;


    vtkPoints *centroids_array = (vtkPoints *) centroids_array_addr;
    vtkIntArray *cell_type_array = (vtkIntArray *) cell_type_array_addr;
    vtkFloatArray *vol_scaling_factors_array = (vtkFloatArray *) vol_scaling_factors_array_addr;
    vtkCharArray *typeArrayRead = (vtkCharArray *) lds->GetPointData()->GetArray("CellType");
    vtkLongArray *idArray = (vtkLongArray *) lds->GetPointData()->GetArray("CellId");

    vector<int> fieldDimVec(3, 0);
    fieldDimVec[0] = fieldDim.x;
    fieldDimVec[1] = fieldDim.y;
    fieldDimVec[2] = fieldDim.z;

    vector<int> pointOrderVec = pointOrder(plane);
    vector<int> dimOrderVec = dimOrder(plane);

    vector<int> dim(3, 0);
    dim[0] = fieldDimVec[dimOrderVec[0]];
    dim[1] = fieldDimVec[dimOrderVec[1]];
    dim[2] = fieldDimVec[dimOrderVec[2]];

    int offset = 0;

    Point3D pt;
    vector<int> ptVec(3, 0);
    //CellG* cell;
    //int type;
    long pc = 0;
    long idxPt;
    long cell_id;
    char cellType;

    //when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned


    for (int j = 0; j < dim[1]; ++j)
        for (int i = 0; i < dim[0]; ++i) {
            ptVec[0] = i;
            ptVec[1] = j;
            ptVec[2] = pos;




            pt.x = ptVec[pointOrderVec[0]];
            pt.y = ptVec[pointOrderVec[1]];
            pt.z = ptVec[pointOrderVec[2]];

            cellType = typeArrayRead->GetValue(indexPoint3D(pt));
            if (!cellType)
                continue;

            idxPt = indexPoint3D(pt);
            cell_id = idArray->GetValue(idxPt);


            cell_id_to_cell_type[cell_id] = (int)cellType;
            cell_id_to_coords_0[cell_id].push_back(i);
            cell_id_to_coords_1[cell_id].push_back(j);
        }

    for(const auto& cell_id_type_pair: cell_id_to_cell_type){
        long cell_id = cell_id_type_pair.first;
        int cell_type = cell_id_type_pair.second;
        const auto & coords_0 = cell_id_to_coords_0[cell_id];
        const auto & coords_1 = cell_id_to_coords_1[cell_id];
        auto vol = coords_0.size()*1.0;
        // in 2D we assume cell-glyph is a sphere
        // so : vol = math.pi*r**2 => r = sqrt(1/math.pi)*sqrt(vol) = 0.564*sqrt(vol)

        vol_scaling_factors_array->InsertNextValue(0.564*pow(vol,0.5));
        cell_type_array->InsertNextValue(cell_type);
        centroids_array->InsertNextPoint(centroid(coords_0), centroid(coords_1), 0.0);



    }

}

void FieldExtractorCML::fillConFieldGlyphs2D(
                std::string con_field_name,
                vtk_obj_addr_int_t centroids_array_addr,
                vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                vtk_obj_addr_int_t scalar_value_at_com_addr,
                std::string plane, int pos){

    // cell_id to cell type map
    unordered_map<long, int> cell_id_to_cell_type;

    unordered_map<long, list<int> > cell_id_to_coords_0;
    unordered_map<long, list<int> > cell_id_to_coords_1;

    auto *conArrayRead = (vtkDoubleArray *) lds->GetPointData()->GetArray(con_field_name.c_str());

    if (!conArrayRead)
        return ;


    auto *centroids_array = (vtkPoints *) centroids_array_addr;
    auto *vol_scaling_factors_array = (vtkFloatArray *) vol_scaling_factors_array_addr;
    auto *typeArrayRead = (vtkCharArray *) lds->GetPointData()->GetArray("CellType");
    auto *idArray = (vtkLongArray *) lds->GetPointData()->GetArray("CellId");
    auto *scalar_value_at_com_array = (vtkFloatArray *) scalar_value_at_com_addr;


    vector<int> fieldDimVec(3, 0);
    fieldDimVec[0] = fieldDim.x;
    fieldDimVec[1] = fieldDim.y;
    fieldDimVec[2] = fieldDim.z;

    vector<int> pointOrderVec = pointOrder(plane);
    vector<int> dimOrderVec = dimOrder(plane);

    vector<int> dim(3, 0);
    dim[0] = fieldDimVec[dimOrderVec[0]];
    dim[1] = fieldDimVec[dimOrderVec[1]];
    dim[2] = fieldDimVec[dimOrderVec[2]];

    int offset = 0;

    Point3D pt;
    vector<int> ptVec(3, 0);
    long pc = 0;
    long idxPt;
    long cell_id;
    char cellType;

    //when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned


    for (int j = 0; j < dim[1]; ++j)
        for (int i = 0; i < dim[0]; ++i) {
            ptVec[0] = i;
            ptVec[1] = j;
            ptVec[2] = pos;




            pt.x = ptVec[pointOrderVec[0]];
            pt.y = ptVec[pointOrderVec[1]];
            pt.z = ptVec[pointOrderVec[2]];

            cellType = typeArrayRead->GetValue(indexPoint3D(pt));
            if (!cellType)
                continue;

            idxPt = indexPoint3D(pt);
            cell_id = idArray->GetValue(idxPt);


            cell_id_to_cell_type[cell_id] = (int)cellType;
            cell_id_to_coords_0[cell_id].push_back(i);
            cell_id_to_coords_1[cell_id].push_back(j);
        }

    vector<int> com(3);
    auto to_xyz_order = permuted_order_to_xyz(plane);
    double con;
    for(const auto& cell_id_type_pair: cell_id_to_cell_type){
        long cell_id = cell_id_type_pair.first;
        int cell_type = cell_id_type_pair.second;
        const auto & coords_0 = cell_id_to_coords_0[cell_id];
        const auto & coords_1 = cell_id_to_coords_1[cell_id];
        auto vol = coords_0.size()*1.0;
        // in 2D we assume cell-glyph is a sphere
        // so : vol = math.pi*r**2 => r = sqrt(1/math.pi)*sqrt(vol) = 0.564*sqrt(vol)

        auto c0 = centroid(coords_0);
        auto c1 = centroid(coords_1);
        double c2 = pos;

        com[0] =  round(c0);
        com[1] =  round(c1);
        com[2] =  round(c2);

        pt.x = (short)com[to_xyz_order[0]];
        pt.y = (short)com[to_xyz_order[1]];
        pt.z = (short)com[to_xyz_order[2]];

        con = conArrayRead->GetValue(indexPoint3D(pt));

        vol_scaling_factors_array->InsertNextValue(0.564*pow(vol,0.5));
        centroids_array->InsertNextPoint(c0, c1, 0.0);
        scalar_value_at_com_array->InsertNextValue(con);

    }


}


void FieldExtractorCML::fillScalarFieldGlyphs2D(
        std::string con_field_name,
        vtk_obj_addr_int_t centroids_array_addr,
        vtk_obj_addr_int_t vol_scaling_factors_array_addr,
        vtk_obj_addr_int_t scalar_value_at_com_addr,
        std::string plane, int pos) {

    fillConFieldGlyphs2D(con_field_name,
                         centroids_array_addr, vol_scaling_factors_array_addr, scalar_value_at_com_addr,
                         plane, pos);

}

void FieldExtractorCML::fillScalarFieldCellLevelGlyphs2D(
                std::string con_field_name,
                vtk_obj_addr_int_t centroids_array_addr,
                vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                vtk_obj_addr_int_t scalar_value_at_com_addr,
                std::string plane, int pos){

    fillConFieldGlyphs2D(con_field_name,
                         centroids_array_addr, vol_scaling_factors_array_addr, scalar_value_at_com_addr,
                         plane, pos);
}


std::vector<int> FieldExtractorCML::fillCellFieldGlyphs3D(vtk_obj_addr_int_t centroids_array_addr,
                                                       vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                                                       vtk_obj_addr_int_t cell_type_array_addr,
                                                       std::vector<int> *types_invisibe_vec,
                                                       bool extractOuterShellOnly){


    // cell_id to cell type map
    unordered_map<long, int> cell_id_to_cell_type;

    unordered_map<long, list<int> > cell_id_to_coords_0;
    unordered_map<long, list<int> > cell_id_to_coords_1;
    unordered_map<long, list<int> > cell_id_to_coords_2;

    unordered_set<int> invisible_types(types_invisibe_vec->begin(), types_invisibe_vec->end());

    unordered_set<int> used_cell_types;

    vtkPoints *centroids_array = (vtkPoints *) centroids_array_addr;
    vtkIntArray *cell_type_array = (vtkIntArray *) cell_type_array_addr;
    vtkFloatArray *vol_scaling_factors_array = (vtkFloatArray *) vol_scaling_factors_array_addr;
    vtkCharArray *typeArrayRead = (vtkCharArray *) lds->GetPointData()->GetArray("CellType");
    vtkLongArray *idArray = (vtkLongArray *) lds->GetPointData()->GetArray("CellId");

    vector<int> fieldDimVec(3, 0);
    fieldDimVec[0] = fieldDim.x;
    fieldDimVec[1] = fieldDim.y;
    fieldDimVec[2] = fieldDim.z;


    int offset = 0;

    Point3D pt;
    vector<int> ptVec(3, 0);
    //CellG* cell;
    //int type;
    long pc = 0;
    long idxPt;
    long cell_id;
    char cellType;



    for (int k = 0; k < fieldDim.z + 2; ++k)
        for (int j = 0; j < fieldDim.y + 2; ++j)
            for (int i = 0; i < fieldDim.x + 2; ++i) {
                if (i == 0 || i == fieldDim.x + 1 || j == 0 || j == fieldDim.y + 1 || k == 0 || k == fieldDim.z + 1) {
                    continue;
                } else {
                    pt.x = i - 1;
                    pt.y = j - 1;
                    pt.z = k - 1;
                    idxPt = indexPoint3D(pt);

                    cellType = typeArrayRead->GetValue(idxPt);
                    if (!cellType)
                        continue;

                    if (invisible_types.find((int)cellType) != invisible_types.end()) continue;

                    idxPt = indexPoint3D(pt);
                    cell_id = idArray->GetValue(idxPt);
                    used_cell_types.insert((int)cellType);

                    cell_id_to_cell_type[cell_id] = (int)cellType;
                    cell_id_to_coords_0[cell_id].push_back(i);
                    cell_id_to_coords_1[cell_id].push_back(j);
                    cell_id_to_coords_2[cell_id].push_back(k);


                }
            }

    vector<double>scaling_factors = {1.0, 1.0, 1.0};

    if (latticeType=="Hexagonal"){
        scaling_factors[0] = 1.0;
        scaling_factors[1] = 0.866;
        scaling_factors[2] = 0.816;
    }
    for(const auto& cell_id_type_pair: cell_id_to_cell_type){
        long cell_id = cell_id_type_pair.first;
        int cell_type = cell_id_type_pair.second;
        const auto & coords_0 = cell_id_to_coords_0[cell_id];
        const auto & coords_1 = cell_id_to_coords_1[cell_id];
        const auto & coords_2 = cell_id_to_coords_2[cell_id];
        auto vol = coords_0.size()*1.0;
        // in 2D we assume cell-glyph is a sphere
        // so : vol = math.pi*r**2 => r = sqrt(1/math.pi)*sqrt(vol) = 0.564*sqrt(vol)

        vol_scaling_factors_array->InsertNextValue(0.62*pow(vol, 0.333));
        cell_type_array->InsertNextValue(cell_type);
        centroids_array->InsertNextPoint(
                scaling_factors[0] * centroid(coords_0),
                scaling_factors[1] * centroid(coords_1),
                scaling_factors[2] * centroid(coords_2));

    }


    return {used_cell_types.begin(), used_cell_types.end()};

}

std::vector<int> FieldExtractorCML::fillConFieldGlyphs3D(std::string con_field_name,
                                      vtk_obj_addr_int_t centroids_array_addr,
                                      vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                                      vtk_obj_addr_int_t scalar_value_at_com_addr,
                                      std::vector<int> *types_invisibe_vec,
                                      bool extractOuterShellOnly){

    // cell_id to cell type map
    unordered_map<long, int> cell_id_to_cell_type;

    unordered_map<long, list<int> > cell_id_to_coords_0;
    unordered_map<long, list<int> > cell_id_to_coords_1;
    unordered_map<long, list<int> > cell_id_to_coords_2;

    auto *conArrayRead = (vtkDoubleArray *) lds->GetPointData()->GetArray(con_field_name.c_str());

    if (!conArrayRead)
        return {};


    auto *centroids_array = (vtkPoints *) centroids_array_addr;
    auto *vol_scaling_factors_array = (vtkFloatArray *) vol_scaling_factors_array_addr;
    auto *typeArrayRead = (vtkCharArray *) lds->GetPointData()->GetArray("CellType");
    auto *idArray = (vtkLongArray *) lds->GetPointData()->GetArray("CellId");
    auto *scalar_value_at_com_array = (vtkFloatArray *) scalar_value_at_com_addr;

    unordered_set<int> invisible_types(types_invisibe_vec->begin(), types_invisibe_vec->end());

    unordered_set<int> used_cell_types;

    vector<int> fieldDimVec(3, 0);
    fieldDimVec[0] = fieldDim.x;
    fieldDimVec[1] = fieldDim.y;
    fieldDimVec[2] = fieldDim.z;


    int offset = 0;

    Point3D pt;
    vector<int> ptVec(3, 0);
    //CellG* cell;
    //int type;
    long pc = 0;
    long idxPt;
    long cell_id;
    char cellType;



    for (int k = 0; k < fieldDim.z + 2; ++k)
        for (int j = 0; j < fieldDim.y + 2; ++j)
            for (int i = 0; i < fieldDim.x + 2; ++i) {
                if (i == 0 || i == fieldDim.x + 1 || j == 0 || j == fieldDim.y + 1 || k == 0 || k == fieldDim.z + 1) {
                    continue;
                } else {
                    pt.x = i - 1;
                    pt.y = j - 1;
                    pt.z = k - 1;
                    idxPt = indexPoint3D(pt);

                    cellType = typeArrayRead->GetValue(idxPt);
                    if (!cellType)
                        continue;

                    if (invisible_types.find((int)cellType) != invisible_types.end()) continue;

                    idxPt = indexPoint3D(pt);
                    cell_id = idArray->GetValue(idxPt);
                    used_cell_types.insert((int)cellType);

                    cell_id_to_cell_type[cell_id] = (int)cellType;
                    cell_id_to_coords_0[cell_id].push_back(i);
                    cell_id_to_coords_1[cell_id].push_back(j);
                    cell_id_to_coords_2[cell_id].push_back(k);


                }
            }

    vector<double>scaling_factors = {1.0, 1.0, 1.0};

    if (latticeType=="Hexagonal"){
        scaling_factors[0] = 1.0;
        scaling_factors[1] = 0.866;
        scaling_factors[2] = 0.816;
    }

    double con;
    for(const auto& cell_id_type_pair: cell_id_to_cell_type){
        long cell_id = cell_id_type_pair.first;
        int cell_type = cell_id_type_pair.second;
        const auto & coords_0 = cell_id_to_coords_0[cell_id];
        const auto & coords_1 = cell_id_to_coords_1[cell_id];
        const auto & coords_2 = cell_id_to_coords_2[cell_id];
        auto vol = coords_0.size()*1.0;
        // in 2D we assume cell-glyph is a sphere
        // so : vol = math.pi*r**2 => r = sqrt(1/math.pi)*sqrt(vol) = 0.564*sqrt(vol)

        vol_scaling_factors_array->InsertNextValue(0.62*pow(vol, 0.333));
        auto c0 = centroid(coords_0);
        auto c1 = centroid(coords_1);
        auto c2 = centroid(coords_2);


        pt.x = (short)round(c0);
        pt.y = (short)round(c1);
        pt.z = (short)round(c2);

        con = conArrayRead->GetValue(indexPoint3D(pt));

        scalar_value_at_com_array->InsertNextValue(con);
        centroids_array->InsertNextPoint(
                scaling_factors[0] * centroid(coords_0),
                scaling_factors[1] * centroid(coords_1),
                scaling_factors[2] * centroid(coords_2));

    }


    return {used_cell_types.begin(), used_cell_types.end()};


}

std::vector<int> FieldExtractorCML::fillScalarFieldGlyphs3D(std::string con_field_name,
                                         vtk_obj_addr_int_t centroids_array_addr,
                                         vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                                         vtk_obj_addr_int_t scalar_value_at_com_addr,
                                         std::vector<int> *types_invisibe_vec,
                                         bool extractOuterShellOnly){
    return fillConFieldGlyphs3D(con_field_name,
            centroids_array_addr,
            vol_scaling_factors_array_addr,
            scalar_value_at_com_addr,
            types_invisibe_vec,
            extractOuterShellOnly);
}

std::vector<int> FieldExtractorCML::fillScalarFieldCellLevelGlyphs3D(std::string con_field_name,
                                                            vtk_obj_addr_int_t centroids_array_addr,
                                                            vtk_obj_addr_int_t vol_scaling_factors_array_addr,
                                                            vtk_obj_addr_int_t scalar_value_at_com_addr,
                                                            std::vector<int> *types_invisibe_vec,
                                                            bool extractOuterShellOnly){
    return fillConFieldGlyphs3D(con_field_name,
                                centroids_array_addr,
                                vol_scaling_factors_array_addr,
                                scalar_value_at_com_addr,
                                types_invisibe_vec,
                                extractOuterShellOnly);
}

bool FieldExtractorCML::fillLinksField2D(
    vtk_obj_addr_int_t points_array_addr, 
    vtk_obj_addr_int_t lines_array_addr, 
    const std::string &plane,
    const int &pos,
    const int &margin
) {
    vector<int> dimOrderVec = dimOrder(plane);
    int dimOrder[] = {dimOrderVec[0], dimOrderVec[1], dimOrderVec[2]};

    Dim3D fieldDim = getFieldDim();
    int fieldDimVec[] = {fieldDim.x, fieldDim.y, fieldDim.z};
    int dim[] = {fieldDimVec[dimOrder[0]], fieldDimVec[dimOrder[1]], fieldDimVec[dimOrder[2]]};

    vtkLongArray *idArray = (vtkLongArray *) lds->GetPointData()->GetArray(FieldWriterCML::CellIdName.c_str());
    vtkLongArray *linksArray = (vtkLongArray *) lds->GetPointData()->GetArray(FieldWriterCML::LinksName.c_str());
    vtkLongArray *linksInternalArray = (vtkLongArray *) lds->GetPointData()->GetArray(FieldWriterCML::LinksInternalName.c_str());
    vtkDoubleArray *anchorsArray = (vtkDoubleArray *) lds->GetPointData()->GetArray(FieldWriterCML::AnchorsName.c_str());

    if(!linksArray || !linksInternalArray || !anchorsArray) 
        return false;
    if(linksArray->GetSize() + linksInternalArray->GetSize() + linksInternalArray->GetSize() == 0) 
        return false;

    int ptCounter = 0;
    vtkPoints *points = (vtkPoints*)points_array_addr;
    vtkCellArray *lines = (vtkCellArray*)lines_array_addr;

    std::unordered_map<long, std::list<int> > coordsx, coordsy, coordsz;
    std::unordered_set<long> coordsids;
    Point3D pt;
    for(int k = 0; k < fieldDim.z + 2; ++k)
        for(int j = 0; j < fieldDim.y + 2; ++j)
            for(int i = 0; i < fieldDim.x + 2; ++i) {
                if(i == 0 || i == fieldDim.x + 1 || j == 0 || j == fieldDim.y + 1 || k == 0 || k == fieldDim.z + 1) 
                    continue;
                else {
                    pt.x = i - 1;
                    pt.y = j - 1;
                    pt.z = k - 1;

                    long cell_id = idArray->GetValue(indexPoint3D(pt));
                    if(!cell_id)
                        continue;

                    coordsids.insert(cell_id);
                    coordsx[cell_id].push_back(i);
                    coordsy[cell_id].push_back(j);
                    coordsz[cell_id].push_back(k);
                }
            }

    std::unordered_map<long, Coordinates3D<double> > centroids;
    for(auto &cell_id : coordsids) 
        centroids[cell_id] = {centroid(coordsx[cell_id]), centroid(coordsy[cell_id]), centroid(coordsz[cell_id])};

    for(int i = 0; i < linksArray->GetSize(); i += 2) {
        long id0 = linksArray->GetValue(i);
        long id1 = linksArray->GetValue(i + 1);

        Coordinates3D<double> _pt0 = centroids[id0];
        Coordinates3D<double> _pt1 = centroids[id1];
        double pt0[] = {_pt0.x, _pt0.y, _pt0.z};
        double pt1[] = {_pt1.x, _pt1.y, _pt1.z};

        vizLinks2D(pt0, pt1, (vtk_obj_addr_int_t)points, (vtk_obj_addr_int_t)lines, dim, dimOrder, pos, margin, ptCounter);
    }
    for(int i = 0; i < linksInternalArray->GetSize(); i += 2) {
        long id0 = linksInternalArray->GetValue(i);
        long id1 = linksInternalArray->GetValue(i + 1);

        Coordinates3D<double> _pt0 = centroids[id0];
        Coordinates3D<double> _pt1 = centroids[id1];
        double pt0[] = {_pt0.x, _pt0.y, _pt0.z};
        double pt1[] = {_pt1.x, _pt1.y, _pt1.z};

        vizLinks2D(pt0, pt1, (vtk_obj_addr_int_t)points, (vtk_obj_addr_int_t)lines, dim, dimOrder, pos, margin, ptCounter);
    }
    for(int i = 0; i < anchorsArray->GetSize(); i++) {
        double *vi = anchorsArray->GetTuple4(i);
        long id0 = (long)vi[0];

        Coordinates3D<double> _pt0 = centroids[id0];
        double pt0[] = {_pt0.x, _pt0.y, _pt0.z};
        double pt1[] = {vi[1], vi[2], vi[3]};

        vizLinks2D(pt0, pt1, (vtk_obj_addr_int_t)points, (vtk_obj_addr_int_t)lines, dim, dimOrder, pos, margin, ptCounter);
    }

    return true;
}

bool FieldExtractorCML::fillLinksField3D(
    vtk_obj_addr_int_t points_array_addr, 
    vtk_obj_addr_int_t lines_array_addr
) {
    Dim3D fieldDim = getFieldDim();
    int fieldDimVec[] = {fieldDim.x, fieldDim.y, fieldDim.z};

    vtkLongArray *idArray = (vtkLongArray *) lds->GetPointData()->GetArray(FieldWriterCML::CellIdName.c_str());
    vtkLongArray *linksArray = (vtkLongArray *) lds->GetPointData()->GetArray(FieldWriterCML::LinksName.c_str());
    vtkLongArray *linksInternalArray = (vtkLongArray *) lds->GetPointData()->GetArray(FieldWriterCML::LinksInternalName.c_str());
    vtkDoubleArray *anchorsArray = (vtkDoubleArray *) lds->GetPointData()->GetArray(FieldWriterCML::AnchorsName.c_str());

    if(!linksArray || !linksInternalArray || !anchorsArray) 
        return false;
    if(linksArray->GetSize() + linksInternalArray->GetSize() + linksInternalArray->GetSize() == 0) 
        return false;

    int ptCounter = 0;
    vtkPoints *points = (vtkPoints*)points_array_addr;
    vtkCellArray *lines = (vtkCellArray*)lines_array_addr;

    std::unordered_map<long, std::list<int> > coordsx, coordsy, coordsz;
    std::unordered_set<long> coordsids;
    Point3D pt;
    for(int k = 0; k < fieldDim.z + 2; ++k)
        for(int j = 0; j < fieldDim.y + 2; ++j)
            for(int i = 0; i < fieldDim.x + 2; ++i) {
                if(i == 0 || i == fieldDim.x + 1 || j == 0 || j == fieldDim.y + 1 || k == 0 || k == fieldDim.z + 1) 
                    continue;
                else {
                    pt.x = i - 1;
                    pt.y = j - 1;
                    pt.z = k - 1;

                    long cell_id = idArray->GetValue(indexPoint3D(pt));
                    if(!cell_id)
                        continue;

                    coordsids.insert(cell_id);
                    coordsx[cell_id].push_back(i);
                    coordsy[cell_id].push_back(j);
                    coordsz[cell_id].push_back(k);
                }
            }

    std::unordered_map<long, Coordinates3D<double> > centroids;
    for(auto &cell_id : coordsids) 
        centroids[cell_id] = {centroid(coordsx[cell_id]), centroid(coordsy[cell_id]), centroid(coordsz[cell_id])};

    for(int i = 0; i < linksArray->GetSize(); i += 2) {
        long id0 = linksArray->GetValue(i);
        long id1 = linksArray->GetValue(i + 1);

        Coordinates3D<double> _pt0 = centroids[id0];
        Coordinates3D<double> _pt1 = centroids[id1];
        double pt0[] = {_pt0.x, _pt0.y, _pt0.z};
        double pt1[] = {_pt1.x, _pt1.y, _pt1.z};

        vizLinks3D(pt0, pt1, (vtk_obj_addr_int_t)points, (vtk_obj_addr_int_t)lines, fieldDimVec, ptCounter);
    }
    for(int i = 0; i < linksInternalArray->GetSize(); i += 2) {
        long id0 = linksInternalArray->GetValue(i);
        long id1 = linksInternalArray->GetValue(i + 1);

        Coordinates3D<double> _pt0 = centroids[id0];
        Coordinates3D<double> _pt1 = centroids[id1];
        double pt0[] = {_pt0.x, _pt0.y, _pt0.z};
        double pt1[] = {_pt1.x, _pt1.y, _pt1.z};

        vizLinks3D(pt0, pt1, (vtk_obj_addr_int_t)points, (vtk_obj_addr_int_t)lines, fieldDimVec, ptCounter);
    }
    for(int i = 0; i < anchorsArray->GetSize(); i++) {
        double *vi = anchorsArray->GetTuple4(i);
        long id0 = (long)vi[0];

        Coordinates3D<double> _pt0 = centroids[id0];
        double pt0[] = {_pt0.x, _pt0.y, _pt0.z};
        double pt1[] = {vi[1], vi[2], vi[3]};

        vizLinks3D(pt0, pt1, (vtk_obj_addr_int_t)points, (vtk_obj_addr_int_t)lines, fieldDimVec, ptCounter);
    }

    return true;
}
