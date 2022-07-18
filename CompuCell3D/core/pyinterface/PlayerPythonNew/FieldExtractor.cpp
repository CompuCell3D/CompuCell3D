    
#include "CellGraphicsData.h"
#include <iostream>
#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>
#include <Utils/Coordinates3D.h>
#include <PublicUtilities/ParallelUtilsOpenMP.h>
#include <vtkIntArray.h>
#include <vtkLongArray.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkType.h>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <unordered_set>
#include <vtkPythonUtil.h>

using namespace std;
using namespace CompuCell3D;


#include "FieldExtractor.h"

FieldExtractor::FieldExtractor():fsPtr(0),potts(0),sim(0)
{
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldExtractor::~FieldExtractor(){

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FieldExtractor::init(Simulator * _sim){
	sim=_sim;
	potts=sim->getPotts();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FieldExtractor::extractCellField(){
	//cerr<<"EXTRACTING CELL FIELD"<<endl;
	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();
	Point3D pt;
	// cerr<< "FIeld Extractor cell field fieldDim="<<fieldDim<<endl;
	CellGraphicsData gd;
	CellG *cell;

	for(pt.x =0 ; pt.x < fieldDim.x ; ++pt.x)
		for(pt.y =0 ; pt.y < fieldDim.y ; ++pt.y)
			for(pt.z =0 ; pt.z < fieldDim.z ; ++pt.z){
				cell=cellFieldG->get(pt);
				if(!cell){
					gd.type=0;
					gd.id=0;
				}else{
					gd.type=cell->type;
					gd.id=cell->id;
				}
				fsPtr->field3DGraphicsData[pt.x][pt.y][pt.z]=gd;
			}
}

void FieldExtractor::fillCellFieldData2D(vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _plane, int _pos){
	vtkIntArray *_cellTypeArray=(vtkIntArray *)_cellTypeArrayAddr;

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

	vector<int> fieldDimVec(3,0);
	fieldDimVec[0]=fieldDim.x;
	fieldDimVec[1]=fieldDim.y;
	fieldDimVec[2]=fieldDim.z;

	vector<int> pointOrderVec=pointOrder(_plane);
	vector<int> dimOrderVec=dimOrder(_plane);

	vector<int> dim(3,0);
	dim[0]=fieldDimVec[dimOrderVec[0]];
	dim[1]=fieldDimVec[dimOrderVec[1]];
	dim[2]=fieldDimVec[dimOrderVec[2]];

    pUtils->setNumberOfWorkNodesAuto();

  int size = (dim[1]+2)*(dim[0]+1);
	_cellTypeArray->SetNumberOfValues(size);
	//For some reasons the points x=0 are eaten up (don't know why).
	//So we just populate empty cellIds.
#pragma omp parallel shared(pointOrderVec, dim, _cellTypeArray, cellFieldG)
  {
#pragma omp for schedule(static) nowait
    for (int i = 0; i < dim[0] + 1; ++i)
    {
      _cellTypeArray->SetValue(i, 0);
    }

    // when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
#pragma omp for schedule(static) nowait
	for(int j =0 ; j<dim[1]+1 ; ++j){
    Point3D pt;
    vector<int> ptVec(3,0);
    CellG* cell;
    int type;

    for(int i =0 ; i<dim[0]+1 ; ++i){
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

			cell=cellFieldG->get(pt);
			if (!cell){
				type=0;
			}else{
				type=cell->type;
			}
      int pos = i + j*(dim[1]+1) + (dim[0]+1);
			_cellTypeArray->SetValue(pos, type);
		}
  }
}
  pUtils->setNumberOfWorkNodesAuto(1);
}

void FieldExtractor::fillCellFieldData2DCartesian(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _cellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane, int _pos) {
    vtkIntArray *_cellTypeArray = (vtkIntArray *)_cellTypeArrayAddr;
    vtkPoints *_pointsArray = (vtkPoints *)_pointsArrayAddr;
    vtkCellArray * _cellsArray = (vtkCellArray*)_cellsArrayAddr;

    Field3D<CellG*> * cellFieldG = potts->getCellFieldG();
    Dim3D fieldDim = cellFieldG->getDim();

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

    //when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
    vector<double*> global_point_vec;
    vtkIdType *_cellsArrayWritePtr;

    pUtils->setNumberOfWorkNodesAuto();;

#pragma omp parallel shared(pointOrderVec, dim, cellFieldG, global_point_vec, _cellsArrayWritePtr)
  {
    vector<double*> local_point_vec;
    Point3D pt;
    vector<int> ptVec(3, 0);
    CellG *cell;
    int type;

#pragma omp for schedule(static,5) nowait
    for (int j = 0; j < dim[1]; ++j){
      for (int i = 0; i < dim[0]; ++i) {
        ptVec[0] = i;
        ptVec[1] = j;
        ptVec[2] = _pos;

        pt.x = ptVec[pointOrderVec[0]];
        pt.y = ptVec[pointOrderVec[1]];
        pt.z = ptVec[pointOrderVec[2]];

        cell = cellFieldG->get(pt);
        if (!cell)
        {
          type = 0;
          continue;
        }
        else
        {
          type = (int) cell->type;
        }

        // notice that we are drawing pixels from other planes on a xy plan so we use ptVec instead of pt. pt is absolute position of the point ptVec is for projection purposes
        Coordinates3D<double> coords(ptVec[0], ptVec[1], 0);
        double* data = new double[9];
        for (int idx = 0; idx < 4; ++idx)
        {
          Coordinates3D<double> cartesianVertex = cartesianVertices[idx] + coords;
          data[idx*2] = cartesianVertex.x;
          data[(idx*2) + 1] = cartesianVertex.y;
        }
        data[8] = type;
        local_point_vec.push_back(data);
      }
    }
  #pragma omp critical
  {
    // https://stackoverflow.com/a/18671256
    // we can force these to be added in-order if we want
    global_point_vec.insert(global_point_vec.end(), local_point_vec.begin(), local_point_vec.end());
  }

#pragma omp barrier
int numPoints = global_point_vec.size();

#pragma omp sections
{
  #pragma omp section
  {
    _cellsArrayWritePtr = _cellsArray->WritePointer(numPoints, numPoints*5);
  }
  #pragma omp section
  {
    _cellTypeArray->SetNumberOfValues(numPoints);
  }
  #pragma omp section
  {
    _pointsArray->SetNumberOfPoints(global_point_vec.size()*4);
  }
}

#pragma omp for schedule(static)
  for (int j = 0; j < numPoints; ++j) {
    int cellPos = j*4;
    auto pts = global_point_vec[j];
    for (int idx = 0; idx < 4; ++idx)
    {
      _pointsArray->SetPoint(cellPos + idx, pts[idx*2], pts[(idx*2)+1], 0.0);
    }

    int arrPos = j*5;
    _cellsArrayWritePtr[arrPos + 0] = 4;
    _cellsArrayWritePtr[arrPos + 1] = cellPos + 0;
    _cellsArrayWritePtr[arrPos + 2] = cellPos + 1;
    _cellsArrayWritePtr[arrPos + 3] = cellPos + 2;
    _cellsArrayWritePtr[arrPos + 4] = cellPos + 3;
    _cellTypeArray->SetValue(j, pts[8]);
    free(pts);
  }
}
  pUtils->setNumberOfWorkNodesAuto(1);
}


void FieldExtractor::fillCellFieldData2DHex(vtk_obj_addr_int_t _cellTypeArrayAddr,vtk_obj_addr_int_t _hexCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane ,  int _pos){
	vtkIntArray *_cellTypeArray=(vtkIntArray *)_cellTypeArrayAddr;
	vtkPoints *_pointsArray=(vtkPoints *)_pointsArrayAddr;
	vtkCellArray * _hexCellsArray=(vtkCellArray*)_hexCellsArrayAddr;

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

	vector<int> fieldDimVec(3,0);
	fieldDimVec[0]=fieldDim.x;
	fieldDimVec[1]=fieldDim.y;
	fieldDimVec[2]=fieldDim.z;

	vector<int> pointOrderVec=pointOrder(_plane);
	vector<int> dimOrderVec=dimOrder(_plane);

	vector<int> dim(3,0);
	dim[0]=fieldDimVec[dimOrderVec[0]];
	dim[1]=fieldDimVec[dimOrderVec[1]];
	dim[2]=fieldDimVec[dimOrderVec[2]];

  vector<std::pair<double, double>> global_point_vec;
  vector<std::pair<double, double>> local_point_vec;

  vector<int> global_type_vec;
  vector<int> local_type_vec;
  vtkIdType *_hexCellsArrayWritePtr;
  int numPoints;
  pUtils->setNumberOfWorkNodesAuto();
  //when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
#pragma omp parallel shared(pointOrderVec, dim, cellFieldG, global_point_vec, global_type_vec, _hexCellsArrayWritePtr, _cellTypeArray, _pointsArray) private(local_point_vec, local_type_vec, numPoints)
  {
#pragma omp for schedule(static,5) nowait
	for(int j =0 ; j<dim[1] ; ++j) {
    Point3D pt;
    vector<int> ptVec(3,0);
    CellG* cell;
    int type=0;
    long pc=0;

		for(int i =0 ; i<dim[0] ; ++i){
      int dataPoint = i + j*dim[1];
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

			cell=cellFieldG->get(pt);
			if (!cell){
				continue;
			}else{
				type=cell->type;
			}

			Coordinates3D<double> hexCoords=HexCoordXY(pt.x,pt.y,pt.z);
			for (int idx = 0; idx < 6; ++idx){
        Coordinates3D<double> hexagonVertex=hexagonVertices[idx]+hexCoords;
        local_point_vec.push_back(std::pair<double, double>(hexagonVertex.x, hexagonVertex.y));
      }

      local_type_vec.push_back(type);
    }
  }
#pragma omp critical
  {
    global_point_vec.insert(global_point_vec.end(), local_point_vec.begin(), local_point_vec.end());
    global_type_vec.insert(global_type_vec.end(), local_type_vec.begin(), local_type_vec.end());
  }

#pragma omp barrier
  numPoints = global_type_vec.size();
  int numHexCellsArray = numPoints * 7;
  int numPointsCount = numPoints * 6;
#pragma omp sections
{
  #pragma omp section
  {
    _hexCellsArrayWritePtr = _hexCellsArray->WritePointer(numPoints, numHexCellsArray);
  }
  #pragma omp section
  {
    _cellTypeArray->SetNumberOfValues(numPoints);
  }
  #pragma omp section
  {
    _pointsArray->SetNumberOfPoints(numPointsCount);
  }
}

#pragma omp for schedule(static)
  for (int j = 0; j < numPoints; ++j)
  {
    // notice that we are drawing pixels from other planes on a xy plan so we use ptVec instead of pt. pt is absolute position of the point ptVec is for projection purposes
    int cellPos = j*6;
    for (int idx = 0; idx < 6; ++idx)
    {
      std::pair<double, double> pt = global_point_vec[cellPos + idx];
      _pointsArray->SetPoint(cellPos + idx, pt.first, pt.second, 0.0);
    }

    int arrPos = j * 7;
    _hexCellsArrayWritePtr[arrPos + 0] = 6;
    _hexCellsArrayWritePtr[arrPos + 1] = cellPos + 0;
    _hexCellsArrayWritePtr[arrPos+2]=cellPos+1;
    _hexCellsArrayWritePtr[arrPos+3]=cellPos+2;
    _hexCellsArrayWritePtr[arrPos+4]=cellPos+3;
    _hexCellsArrayWritePtr[arrPos+5]=cellPos+4;
    _hexCellsArrayWritePtr[arrPos+6]=cellPos+5;
    _cellTypeArray->SetValue(j, global_type_vec[j]);
  }
}
  pUtils->setNumberOfWorkNodesAuto();
}

void FieldExtractor::fillBorderData2D(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr, std::string _plane, int _pos)
{
  vtkPoints *points = (vtkPoints *)_pointArrayAddr;
  vtkCellArray *lines = (vtkCellArray *)_linesArrayAddr;

  Field3D<CellG *> *cellFieldG = potts->getCellFieldG();
  Dim3D fieldDim = cellFieldG->getDim();

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

  pUtils->setNumberOfWorkNodesAuto();
  vector<std::tuple<double, double, double, double>> global_points;
  vtkIdType *linesWritePtr;
#pragma omp parallel shared(pointOrderVec, dim, cellFieldG, points, lines, global_points, linesWritePtr)
  {
    vector<std::tuple<double, double, double, double>> local_points;
#pragma omp for schedule(static) nowait
    for (int i = 0; i < dim[0]; ++i)
    {
      Point3D pt;
      vector<int> ptVec(3, 0);
      Point3D ptN;
      vector<int> ptNVec(3, 0);

      for (int j = 0; j < dim[1]; ++j)
      {
        ptVec[0] = i;
        ptVec[1] = j;
        ptVec[2] = _pos;

        pt.x = ptVec[pointOrderVec[0]];
        pt.y = ptVec[pointOrderVec[1]];
        pt.z = ptVec[pointOrderVec[2]];

        if (i > 0 && j < dim[1])
        {
          ptNVec[0] = i - 1;
          ptNVec[1] = j;
          ptNVec[2] = _pos;
          ptN.x = ptNVec[pointOrderVec[0]];
          ptN.y = ptNVec[pointOrderVec[1]];
          ptN.z = ptNVec[pointOrderVec[2]];
          if (cellFieldG->get(pt) != cellFieldG->get(ptN))
          {
            local_points.push_back(make_tuple<double, double>((double)i, (double)j, (double)i, (double)j + 1));
            // local_points.push_back(std::pair<double, double>(i, j + 1));
          }
        }
        if (j > 0 && i < dim[0])
        {
          ptNVec[0] = i;
          ptNVec[1] = j - 1;
          ptNVec[2] = _pos;
          ptN.x = ptNVec[pointOrderVec[0]];
          ptN.y = ptNVec[pointOrderVec[1]];
          ptN.z = ptNVec[pointOrderVec[2]];
          if (cellFieldG->get(pt) != cellFieldG->get(ptN))
          {
            local_points.push_back(make_tuple<double, double>((double)i, (double)j, (double)i + 1, (double)j));
            // local_points.push_back(std::pair<double, double>(i + 1, j));
          }
        }

        if (i < dim[0] && j < dim[1])
        {
          ptNVec[0] = i + 1;
          ptNVec[1] = j;
          ptNVec[2] = _pos;
          ptN.x = ptNVec[pointOrderVec[0]];
          ptN.y = ptNVec[pointOrderVec[1]];
          ptN.z = ptNVec[pointOrderVec[2]];
          if (cellFieldG->get(pt) != cellFieldG->get(ptN))
          {
            local_points.push_back(make_tuple<double, double>((double)i + 1, (double)j, (double)i + 1, (double)j + 1));
            // local_points.push_back(std::pair<double, double>(i + 1, j + 1));
          }
        }

        if (i < dim[0] && j < dim[1])
        {
          ptNVec[0] = i;
          ptNVec[1] = j + 1;
          ptNVec[2] = _pos;
          ptN.x = ptNVec[pointOrderVec[0]];
          ptN.y = ptNVec[pointOrderVec[1]];
          ptN.z = ptNVec[pointOrderVec[2]];
          if (cellFieldG->get(pt) != cellFieldG->get(ptN))
          {
            local_points.push_back(make_tuple<double, double>((double)i, (double)j + 1, (double)i + 1, (double)j + 1));
            // local_points.push_back(std::pair<double, double>(i + 1, j + 1));
          }
        }
      }
    }
#pragma omp critical
    {
      global_points.insert(global_points.end(), local_points.begin(), local_points.end());
    }

#pragma omp barrier

#pragma omp sections
    {
#pragma omp section
      {
        linesWritePtr = lines->WritePointer(global_points.size(), global_points.size() * 3);
      }
#pragma omp section
      {
        points->SetNumberOfPoints(global_points.size()*2);
      }
    }

    int pc, pt_pos = 0;
#pragma omp for schedule(static)
    for (int j = 0; j < global_points.size(); ++j)
    {
      std::tuple<double, double, double, double> pt = global_points[j];
      pt_pos = j*2;
      points->SetPoint(pt_pos, std::get<0>(pt), std::get<1>(pt), 0);
      points->SetPoint(pt_pos + 1, std::get<2>(pt), std::get<3>(pt), 0);
      pc = j * 3;
      linesWritePtr[pc] = 2;
      linesWritePtr[pc + 1] = pt_pos;
      linesWritePtr[pc + 2] = pt_pos + 1;
    }
  }
  pUtils->setNumberOfWorkNodesAuto(1);
}

void FieldExtractor::fillBorderData2DHex(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr, std::string _plane, int _pos)
{
  //this function can be shortened but for now I am leaving it the way it is
	vtkPoints *points = (vtkPoints *)_pointArrayAddr;
	vtkCellArray * lines=(vtkCellArray *)_linesArrayAddr;

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

	vector<int> fieldDimVec(3,0);
	fieldDimVec[0]=fieldDim.x;
	fieldDimVec[1]=fieldDim.y;
	fieldDimVec[2]=fieldDim.z;

	vector<int> pointOrderVec=pointOrder(_plane);
	vector<int> dimOrderVec=dimOrder(_plane);

	vector<int> dim(3,0);
	dim[0]=fieldDimVec[dimOrderVec[0]];
	dim[1]=fieldDimVec[dimOrderVec[1]];
	dim[2]=fieldDimVec[dimOrderVec[2]];

  vector<std::pair<double, double>> global_points;
  vtkIdType *linesWritePtr;
  pUtils->setNumberOfWorkNodesAuto();
#pragma omp parallel shared(pointOrderVec, dim, cellFieldG, points, lines, global_points, linesWritePtr)
  {

  vector<std::pair<double,double>> local_points;
  Point3D pt;
  vector<int> ptVec(3, 0);
  Point3D ptN;
  vector<int> ptNVec(3, 0);
#pragma omp for schedule(static, 5) nowait
  for(int i=0; i <dim[0]; ++i) {
		for(int j=0; j <dim[1]; ++j){
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];
			Coordinates3D<double> hexCoords=HexCoordXY(pt.x,pt.y,pt.z);
      if (pt.z%3==0){ // z divisible by 3
        if(pt.y%2){ //y_odd
            if(pt.x-1>=0){
                ptN.x=pt.x-1;
                ptN.y=pt.y;
                ptN.z=pt.z;
                if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                    Coordinates3D<double> hexCoordsP1=hexagonVertices[4]+hexCoords;
                    Coordinates3D<double> hexCoordsP2=hexagonVertices[5]+hexCoords;
                    local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                    local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                }
            }
            if(pt.x-1>=0 && pt.y+1<dim[1]){
                ptN.x=pt.x-1;
                ptN.y=pt.y+1;
                ptN.z=pt.z;
                if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                    Coordinates3D<double> hexCoordsP1=hexagonVertices[5]+hexCoords;
                    Coordinates3D<double> hexCoordsP2=hexagonVertices[0]+hexCoords;
                    local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                    local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                }
            }
            if( pt.y+1<dim[1]){
                ptN.x=pt.x;
                ptN.y=pt.y+1;
                ptN.z=pt.z;
                if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                    Coordinates3D<double> hexCoordsP1=hexagonVertices[0]+hexCoords;
                    Coordinates3D<double> hexCoordsP2=hexagonVertices[1]+hexCoords;
                    local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                    local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                }
            }
            if( pt.x+1<dim[0]){
                ptN.x=pt.x+1;
                ptN.y=pt.y;
                ptN.z=pt.z;
                if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                    Coordinates3D<double> hexCoordsP1=hexagonVertices[1]+hexCoords;
                    Coordinates3D<double> hexCoordsP2=hexagonVertices[2]+hexCoords;
                    local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                    local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                }
            }
            if( pt.y-1>=0){
                ptN.x=pt.x;
                ptN.y=pt.y-1;
                ptN.z=pt.z;
                if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                    Coordinates3D<double> hexCoordsP1=hexagonVertices[2]+hexCoords;
                    Coordinates3D<double> hexCoordsP2=hexagonVertices[3]+hexCoords;
                    local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                    local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                }
            }
            if(pt.x-1>=0 && pt.y-1>=0){
                ptN.x=pt.x-1;
                ptN.y=pt.y-1;
                ptN.z=pt.z;
                if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                    Coordinates3D<double> hexCoordsP1=hexagonVertices[3]+hexCoords;
                    Coordinates3D<double> hexCoordsP2=hexagonVertices[4]+hexCoords;
                    local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                    local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                }
            }

        }else{//y_even
          if(pt.x-1>=0){
                  ptN.x=pt.x-1;
                  ptN.y=pt.y;
                  ptN.z=pt.z;
                  if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                      Coordinates3D<double> hexCoordsP1=hexagonVertices[4]+hexCoords;
                      Coordinates3D<double> hexCoordsP2=hexagonVertices[5]+hexCoords;
                      local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                      local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                  }
              }
            if( pt.y+1<dim[1]){
                  ptN.x=pt.x;
                  ptN.y=pt.y+1;
                  ptN.z=pt.z;
                  if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                      Coordinates3D<double> hexCoordsP1=hexagonVertices[5]+hexCoords;
                      Coordinates3D<double> hexCoordsP2=hexagonVertices[0]+hexCoords;
                      local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                      local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                  }
              }
            if(pt.x+1<dim[0] && pt.y+1<dim[1]){
                ptN.x=pt.x+1;
                ptN.y=pt.y+1;
                ptN.z=pt.z;
                if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                    Coordinates3D<double> hexCoordsP1=hexagonVertices[0]+hexCoords;
                    Coordinates3D<double> hexCoordsP2=hexagonVertices[1]+hexCoords;
                    local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                    local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                }
            }
            if(pt.x+1<dim[0] ){
                ptN.x=pt.x+1;
                ptN.y=pt.y;
                ptN.z=pt.z;
                if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                    Coordinates3D<double> hexCoordsP1=hexagonVertices[1]+hexCoords;
                    Coordinates3D<double> hexCoordsP2=hexagonVertices[2]+hexCoords;
                    local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                    local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                }
            }
            if(pt.x+1<dim[0] && pt.y-1>=0 ){
                ptN.x=pt.x+1;
                ptN.y=pt.y-1;
                ptN.z=pt.z;
                if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                    Coordinates3D<double> hexCoordsP1=hexagonVertices[2]+hexCoords;
                    Coordinates3D<double> hexCoordsP2=hexagonVertices[3]+hexCoords;
                    local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                    local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                }
            }
            if( pt.y-1>=0 ){
                ptN.x=pt.x;
                ptN.y=pt.y-1;
                ptN.z=pt.z;
                if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                    Coordinates3D<double> hexCoordsP1=hexagonVertices[3]+hexCoords;
                    Coordinates3D<double> hexCoordsP2=hexagonVertices[4]+hexCoords;
                    local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                    local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                }
            }
          }
      }

      else { //apparently for  pt.z%3==1 and pt.z%3==2 xy hex shifts are the same so one code serves them both
        if(pt.y%2){ //y_odd
            if(pt.x-1>=0){
                ptN.x=pt.x-1;
                ptN.y=pt.y;
                ptN.z=pt.z;
                if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                    Coordinates3D<double> hexCoordsP1=hexagonVertices[4]+hexCoords;
                    Coordinates3D<double> hexCoordsP2=hexagonVertices[5]+hexCoords;
                    local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                    local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                }
            }
            if(pt.x-1>=0 && pt.y+1<dim[1]){
                // ptN.x=pt.x-1;
                // ptN.y=pt.y+1;
                ptN.x=pt.x+1;
                ptN.y=pt.y-1;

                ptN.z=pt.z;
                if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                    Coordinates3D<double> hexCoordsP1=hexagonVertices[2]+hexCoords;
                    Coordinates3D<double> hexCoordsP2=hexagonVertices[3]+hexCoords;
                    local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                    local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                }
            }
            if( pt.y+1<dim[1]){
                ptN.x=pt.x;
                ptN.y=pt.y+1;
                ptN.z=pt.z;
                if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                    Coordinates3D<double> hexCoordsP1=hexagonVertices[5]+hexCoords;
                    Coordinates3D<double> hexCoordsP2=hexagonVertices[0]+hexCoords;
                    local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                    local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                }
            }
            if( pt.x+1<dim[0]){
                ptN.x=pt.x+1;
                ptN.y=pt.y;
                ptN.z=pt.z;
                if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                    Coordinates3D<double> hexCoordsP1=hexagonVertices[1]+hexCoords;
                    Coordinates3D<double> hexCoordsP2=hexagonVertices[2]+hexCoords;
                    local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                    local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                }
            }
            if( pt.y-1>=0){
                ptN.x=pt.x;
                ptN.y=pt.y-1;
                ptN.z=pt.z;
                if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                    Coordinates3D<double> hexCoordsP1=hexagonVertices[3]+hexCoords;
                    Coordinates3D<double> hexCoordsP2=hexagonVertices[4]+hexCoords;
                    local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                    local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                }
            }
            if(pt.x-1>=0 && pt.y-1>=0){
                ptN.x=pt.x+1;
                ptN.y=pt.y+1;
                ptN.z=pt.z;
                if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                    Coordinates3D<double> hexCoordsP1=hexagonVertices[0]+hexCoords;
                    Coordinates3D<double> hexCoordsP2=hexagonVertices[1]+hexCoords;
                    local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                    local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                }
            }

        }else{//y_even
              if(pt.x-1>=0){
                  ptN.x=pt.x-1;
                  ptN.y=pt.y;
                  ptN.z=pt.z;
                  if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                      Coordinates3D<double> hexCoordsP1=hexagonVertices[4]+hexCoords;
                      Coordinates3D<double> hexCoordsP2=hexagonVertices[5]+hexCoords;
                      local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                      local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                  }
              }
              if(pt.x-1>=0 && pt.y+1<dim[1]){
                  // ptN.x=pt.x-1;
                  // ptN.y=pt.y+1;
                  ptN.x=pt.x-1;
                  ptN.y=pt.y+1;

                  ptN.z=pt.z;
                  if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                      Coordinates3D<double> hexCoordsP1=hexagonVertices[5]+hexCoords;
                      Coordinates3D<double> hexCoordsP2=hexagonVertices[0]+hexCoords;
                      local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                      local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                  }
              }
              if( pt.y+1<dim[1]){
                  ptN.x=pt.x;
                  ptN.y=pt.y+1;
                  ptN.z=pt.z;
                  if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                      Coordinates3D<double> hexCoordsP1=hexagonVertices[0]+hexCoords;
                      Coordinates3D<double> hexCoordsP2=hexagonVertices[1]+hexCoords;
                      local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                      local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                  }
              }
              if( pt.x+1<dim[0]){
                  ptN.x=pt.x+1;
                  ptN.y=pt.y;
                  ptN.z=pt.z;
                  if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                      Coordinates3D<double> hexCoordsP1=hexagonVertices[1]+hexCoords;
                      Coordinates3D<double> hexCoordsP2=hexagonVertices[2]+hexCoords;
                      local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                      local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                  }
              }
              if( pt.y-1>=0){
                  ptN.x=pt.x;
                  ptN.y=pt.y-1;
                  ptN.z=pt.z;
                  if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                      Coordinates3D<double> hexCoordsP1=hexagonVertices[2]+hexCoords;
                      Coordinates3D<double> hexCoordsP2=hexagonVertices[3]+hexCoords;
                      local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                      local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                  }
              }
              if(pt.x-1>=0 && pt.y-1>=0){
                  ptN.x=pt.x-1;
                  ptN.y=pt.y-1;
                  ptN.z=pt.z;
                  if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                      Coordinates3D<double> hexCoordsP1=hexagonVertices[3]+hexCoords;
                      Coordinates3D<double> hexCoordsP2=hexagonVertices[4]+hexCoords;
                      local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                      local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                  }
              }
          }
      }
		}
  }
#pragma omp critical
  {
    global_points.insert(global_points.end(), local_points.begin(), local_points.end());
  }

#pragma omp barrier

#pragma omp sections
{
#pragma omp section
  {
    linesWritePtr = lines->WritePointer(global_points.size()/2, (global_points.size()/2) * 3);
  }
#pragma omp section
  {
    points->SetNumberOfPoints(global_points.size());
  }
}
#pragma omp for schedule(static)
  for (int j = 0; j < global_points.size(); j+=2)
  {
    std::pair<double, double> pt1 = global_points[j];
    std::pair<double, double> pt2 = global_points[j + 1];
    points->SetPoint(j, pt1.first, pt1.second, 0);
    points->SetPoint(j + 1, pt2.first, pt2.second, 0);
    int pc = (j / 2) * 3;
    linesWritePtr[pc] = 2;
    linesWritePtr[pc + 1] = j;
    linesWritePtr[pc + 2] = j + 1;
  }
}
pUtils->setNumberOfWorkNodesAuto(1);
}

void FieldExtractor::fillClusterBorderData2D(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr, std::string _plane, int _pos)
{
  vtkPoints *points = (vtkPoints *)_pointArrayAddr;
  vtkCellArray *lines = (vtkCellArray *)_linesArrayAddr;

  Field3D<CellG *> *cellFieldG = potts->getCellFieldG();
  Dim3D fieldDim = cellFieldG->getDim();

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

  vector<std::pair<double, double>> global_points;
  vtkIdType *linesWritePtr;

  pUtils->setNumberOfWorkNodesAuto();
#pragma omp parallel shared(pointOrderVec, dim, cellFieldG, points, lines, global_points, linesWritePtr)
{
  vector<std::pair<double,double>> local_points;
  Point3D pt;
  vector<int> ptVec(3, 0);
  Point3D ptN;
  vector<int> ptNVec(3, 0);
#pragma omp for schedule(static, 5) nowait
  for (int i = 0; i < dim[0]; ++i) {
    for (int j = 0; j < dim[1]; ++j) {
      ptVec[0] = i;
      ptVec[1] = j;
      ptVec[2] = _pos;

      pt.x = ptVec[pointOrderVec[0]];
      pt.y = ptVec[pointOrderVec[1]];
      pt.z = ptVec[pointOrderVec[2]];

      if (cellFieldG->get(pt) == 0)
        continue;

      long clusterId = cellFieldG->get(pt)->clusterId;

      if (i > 0 && j < dim[1])
      {
        ptNVec[0] = i - 1;
        ptNVec[1] = j;
        ptNVec[2] = _pos;
        ptN.x = ptNVec[pointOrderVec[0]];
        ptN.y = ptNVec[pointOrderVec[1]];
        ptN.z = ptNVec[pointOrderVec[2]];
        if ((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId)
        {
          local_points.push_back({i,j});
          local_points.push_back({i, j + 1});
        }
      }
      if (j > 0 && i < dim[0])
      {
        ptNVec[0] = i;
        ptNVec[1] = j - 1;
        ptNVec[2] = _pos;
        ptN.x = ptNVec[pointOrderVec[0]];
        ptN.y = ptNVec[pointOrderVec[1]];
        ptN.z = ptNVec[pointOrderVec[2]];
        if ((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId)
        {
          local_points.push_back({i, j});
          local_points.push_back({i + 1, j});
        }
      }

      if (i < dim[0] && j < dim[1])
      {
        ptNVec[0] = i + 1;
        ptNVec[1] = j;
        ptNVec[2] = _pos;
        ptN.x = ptNVec[pointOrderVec[0]];
        ptN.y = ptNVec[pointOrderVec[1]];
        ptN.z = ptNVec[pointOrderVec[2]];
        if ((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId)
        {
          local_points.push_back({i + 1, j});
          local_points.push_back({i + 1, j + 1});
        }
      }

      if (i < dim[0] && j < dim[1])
      {
        ptNVec[0] = i;
        ptNVec[1] = j + 1;
        ptNVec[2] = _pos;
        ptN.x = ptNVec[pointOrderVec[0]];
        ptN.y = ptNVec[pointOrderVec[1]];
        ptN.z = ptNVec[pointOrderVec[2]];
        if ((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId)
        {
          local_points.push_back({i, j + 1});
          local_points.push_back({i + 1, j + 1});
        }
      }
    }
  }
#pragma omp critical
  {
    global_points.insert(global_points.end(), local_points.begin(), local_points.end());
  }
#pragma omp barrier

#pragma omp sections
  {
#pragma omp section
    {
      linesWritePtr = lines->WritePointer(global_points.size() / 2, (global_points.size() / 2) * 3);
    }
#pragma omp section
    {
      points->SetNumberOfPoints(global_points.size());
    }
  }
#pragma omp for schedule(static)
  for (int j = 0; j < global_points.size(); j += 2)
  {
    std::pair<double, double> pt1 = global_points[j];
    std::pair<double, double> pt2 = global_points[j + 1];
    points->SetPoint(j, pt1.first, pt1.second, 0);
    points->SetPoint(j + 1, pt2.first, pt2.second, 0);
    int pc = (j / 2) * 3;
    linesWritePtr[pc] = 2;
    linesWritePtr[pc + 1] = j;
    linesWritePtr[pc + 2] = j + 1;
  }
}
pUtils->setNumberOfWorkNodesAuto(1);
}

void FieldExtractor::fillClusterBorderData2DHex(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr, std::string _plane, int _pos)
{
  //this function has to be redone in the same spirit as fillBorderData2DHex
	vtkPoints *points = (vtkPoints *)_pointArrayAddr;
	vtkCellArray * lines = (vtkCellArray *)_linesArrayAddr;

	Field3D<CellG*> * cellFieldG = potts->getCellFieldG();
	Dim3D fieldDim = cellFieldG->getDim();

	vector<int> fieldDimVec(3,0);
	fieldDimVec[0]=fieldDim.x;
	fieldDimVec[1]=fieldDim.y;
	fieldDimVec[2]=fieldDim.z;

	vector<int> pointOrderVec=pointOrder(_plane);
	vector<int> dimOrderVec=dimOrder(_plane);

	vector<int> dim(3,0);
	dim[0]=fieldDimVec[dimOrderVec[0]];
	dim[1]=fieldDimVec[dimOrderVec[1]];
	dim[2]=fieldDimVec[dimOrderVec[2]];

  vector<std::pair<double, double>> global_points;
  vtkIdType *linesWritePtr;
  pUtils->setNumberOfWorkNodesAuto();
#pragma omp parallel shared(pointOrderVec, dim, cellFieldG, points, lines, global_points, linesWritePtr)
{
  Point3D pt;
	vector<int> ptVec(3,0);
	Point3D ptN;
	vector<int> ptNVec(3,0);
  vector<std::pair<double, double>> local_points;

#pragma omp for schedule(static, 5) nowait
  for(int i=0; i <dim[0]; ++i) {
		for(int j=0; j <dim[1]; ++j){
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];
			Coordinates3D<double> hexCoords=HexCoordXY(pt.x,pt.y,pt.z);

			if (cellFieldG->get(pt) == 0) continue;
			long clusterId = cellFieldG->get(pt)->clusterId;
      if (pt.z%3==0){ // z divisible by 3
        if(pt.y%2){ //y_odd
          if(pt.x-1>=0){
              ptN.x=pt.x-1;
              ptN.y=pt.y;
              ptN.z=pt.z;
              if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                  Coordinates3D<double> hexCoordsP1=hexagonVertices[4]+hexCoords;
                  Coordinates3D<double> hexCoordsP2=hexagonVertices[5]+hexCoords;
                  local_points.push_back(std::pair<double, double>(hexCoordsP1.x, hexCoordsP1.y));
                  local_points.push_back(std::pair<double, double>(hexCoordsP2.x, hexCoordsP2.y));
              }
          }
          if(pt.x-1>=0 && pt.y+1<dim[1]){
              ptN.x=pt.x-1;
              ptN.y=pt.y+1;
              ptN.z=pt.z;
              if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                  Coordinates3D<double> hexCoordsP1=hexagonVertices[5]+hexCoords;
                  Coordinates3D<double> hexCoordsP2=hexagonVertices[0]+hexCoords;
                  local_points.push_back(std::pair<double, double>(hexCoordsP1.x, hexCoordsP1.y));
                  local_points.push_back(std::pair<double, double>(hexCoordsP2.x, hexCoordsP2.y));
              }
          }
          if( pt.y+1<dim[1]){
              ptN.x=pt.x;
              ptN.y=pt.y+1;
              ptN.z=pt.z;
              if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                  Coordinates3D<double> hexCoordsP1=hexagonVertices[0]+hexCoords;
                  Coordinates3D<double> hexCoordsP2=hexagonVertices[1]+hexCoords;
                  local_points.push_back(std::pair<double, double>(hexCoordsP1.x, hexCoordsP1.y));
                  local_points.push_back(std::pair<double, double>(hexCoordsP2.x, hexCoordsP2.y));
              }
          }
          if( pt.x+1<dim[0]){
              ptN.x=pt.x+1;
              ptN.y=pt.y;
              ptN.z=pt.z;
              if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                  Coordinates3D<double> hexCoordsP1=hexagonVertices[1]+hexCoords;
                  Coordinates3D<double> hexCoordsP2=hexagonVertices[2]+hexCoords;
                  local_points.push_back(std::pair<double, double>(hexCoordsP1.x, hexCoordsP1.y));
                  local_points.push_back(std::pair<double, double>(hexCoordsP2.x, hexCoordsP2.y));
              }
          }
          if( pt.y-1>=0){
              ptN.x=pt.x;
              ptN.y=pt.y-1;
              ptN.z=pt.z;
              if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                  Coordinates3D<double> hexCoordsP1=hexagonVertices[2]+hexCoords;
                  Coordinates3D<double> hexCoordsP2=hexagonVertices[3]+hexCoords;
                    local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                    local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));

              }
          }
          if(pt.x-1>=0 && pt.y-1>=0){
              ptN.x=pt.x-1;
              ptN.y=pt.y-1;
              ptN.z=pt.z;
              if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                  Coordinates3D<double> hexCoordsP1=hexagonVertices[3]+hexCoords;
                  Coordinates3D<double> hexCoordsP2=hexagonVertices[4]+hexCoords;
                  local_points.push_back(std::pair<double, double>(hexCoordsP1.x, hexCoordsP1.y));
                  local_points.push_back(std::pair<double, double>(hexCoordsP2.x, hexCoordsP2.y));
              }
          }
        }else{//y_even
          if(pt.x-1>=0){
              ptN.x=pt.x-1;
              ptN.y=pt.y;
              ptN.z=pt.z;
              if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                  Coordinates3D<double> hexCoordsP1=hexagonVertices[4]+hexCoords;
                  Coordinates3D<double> hexCoordsP2=hexagonVertices[5]+hexCoords;
                  local_points.push_back(std::pair<double, double>(hexCoordsP1.x, hexCoordsP1.y));
                  local_points.push_back(std::pair<double, double>(hexCoordsP2.x, hexCoordsP2.y));
              }
          }
          if( pt.y+1<dim[1]){
              ptN.x=pt.x;
              ptN.y=pt.y+1;
              ptN.z=pt.z;
              if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                  Coordinates3D<double> hexCoordsP1=hexagonVertices[5]+hexCoords;
                  Coordinates3D<double> hexCoordsP2=hexagonVertices[0]+hexCoords;
                  local_points.push_back(std::pair<double, double>(hexCoordsP1.x, hexCoordsP1.y));
                  local_points.push_back(std::pair<double, double>(hexCoordsP2.x, hexCoordsP2.y));
              }
          }
          if(pt.x+1<dim[0] && pt.y+1<dim[1]){
              ptN.x=pt.x+1;
              ptN.y=pt.y+1;
              ptN.z=pt.z;
              if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                  Coordinates3D<double> hexCoordsP1=hexagonVertices[0]+hexCoords;
                  Coordinates3D<double> hexCoordsP2=hexagonVertices[1]+hexCoords;
                  local_points.push_back(std::pair<double, double>(hexCoordsP1.x, hexCoordsP1.y));
                  local_points.push_back(std::pair<double, double>(hexCoordsP2.x, hexCoordsP2.y));
              }
          }
          if(pt.x+1<dim[0] ){
              ptN.x=pt.x+1;
              ptN.y=pt.y;
              ptN.z=pt.z;
              if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                  Coordinates3D<double> hexCoordsP1=hexagonVertices[1]+hexCoords;
                  Coordinates3D<double> hexCoordsP2=hexagonVertices[2]+hexCoords;
                  local_points.push_back(std::pair<double, double>(hexCoordsP1.x, hexCoordsP1.y));
                  local_points.push_back(std::pair<double, double>(hexCoordsP2.x, hexCoordsP2.y));
              }
          }
          if(pt.x+1<dim[0] && pt.y-1>=0 ){
              ptN.x=pt.x+1;
              ptN.y=pt.y-1;
              ptN.z=pt.z;
              if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                  Coordinates3D<double> hexCoordsP1=hexagonVertices[2]+hexCoords;
                  Coordinates3D<double> hexCoordsP2=hexagonVertices[3]+hexCoords;
                  local_points.push_back(std::pair<double, double>(hexCoordsP1.x, hexCoordsP1.y));
                  local_points.push_back(std::pair<double, double>(hexCoordsP2.x, hexCoordsP2.y));
              }
          }
          if( pt.y-1>=0 ){
              ptN.x=pt.x;
              ptN.y=pt.y-1;
              ptN.z=pt.z;
              if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                  Coordinates3D<double> hexCoordsP1=hexagonVertices[3]+hexCoords;
                  Coordinates3D<double> hexCoordsP2=hexagonVertices[4]+hexCoords;
                  local_points.push_back(std::pair<double, double>(hexCoordsP1.x, hexCoordsP1.y));
                  local_points.push_back(std::pair<double, double>(hexCoordsP2.x, hexCoordsP2.y));
              }
          }
        }
      }else{//apparently for  pt.z%3==1 and pt.z%3==2 xy hex shifts are the same so one code serves them both
        if(pt.y%2){ //y_odd
          if(pt.x-1>=0){
              ptN.x=pt.x-1;
              ptN.y=pt.y;
              ptN.z=pt.z;

              if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                  Coordinates3D<double> hexCoordsP1=hexagonVertices[4]+hexCoords;
                  Coordinates3D<double> hexCoordsP2=hexagonVertices[5]+hexCoords;
                  local_points.push_back(std::pair<double, double>(hexCoordsP1.x, hexCoordsP1.y));
                  local_points.push_back(std::pair<double, double>(hexCoordsP2.x, hexCoordsP2.y));
              }
          }
          if(pt.x-1>=0 && pt.y+1<dim[1]){
              ptN.x=pt.x+1;
              ptN.y=pt.y-1;

              ptN.z=pt.z;
              if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                  Coordinates3D<double> hexCoordsP1=hexagonVertices[2]+hexCoords;
                  Coordinates3D<double> hexCoordsP2=hexagonVertices[3]+hexCoords;
                    local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                    local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));

              }
          }
          if( pt.y+1<dim[1]){
              ptN.x=pt.x;
              ptN.y=pt.y+1;
              ptN.z=pt.z;
              if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                  Coordinates3D<double> hexCoordsP1=hexagonVertices[5]+hexCoords;
                  Coordinates3D<double> hexCoordsP2=hexagonVertices[0]+hexCoords;
                  local_points.push_back(std::pair<double, double>(hexCoordsP1.x, hexCoordsP1.y));
                  local_points.push_back(std::pair<double, double>(hexCoordsP2.x, hexCoordsP2.y));
              }
          }
          if( pt.x+1<dim[0]){
              ptN.x=pt.x+1;
              ptN.y=pt.y;
              ptN.z=pt.z;
              if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                  Coordinates3D<double> hexCoordsP1=hexagonVertices[1]+hexCoords;
                  Coordinates3D<double> hexCoordsP2=hexagonVertices[2]+hexCoords;
                  local_points.push_back(std::pair<double, double>(hexCoordsP1.x, hexCoordsP1.y));
                  local_points.push_back(std::pair<double, double>(hexCoordsP2.x, hexCoordsP2.y));
              }
          }
          if( pt.y-1>=0){
              ptN.x=pt.x;
              ptN.y=pt.y-1;
              ptN.z=pt.z;
              if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                  Coordinates3D<double> hexCoordsP1=hexagonVertices[3]+hexCoords;
                  Coordinates3D<double> hexCoordsP2=hexagonVertices[4]+hexCoords;
                    local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                    local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));

              }
          }
          if(pt.x-1>=0 && pt.y-1>=0){
                ptN.x=pt.x+1;
                ptN.y=pt.y+1;
                ptN.z=pt.z;
                if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                    Coordinates3D<double> hexCoordsP1=hexagonVertices[0]+hexCoords;
                    Coordinates3D<double> hexCoordsP2=hexagonVertices[1]+hexCoords;
                    local_points.push_back(std::pair<double, double>(hexCoordsP1.x, hexCoordsP1.y));
                    local_points.push_back(std::pair<double, double>(hexCoordsP2.x, hexCoordsP2.y));
                }
            }
        }else{ //yeven
          if(pt.x-1>=0){
              ptN.x=pt.x-1;
              ptN.y=pt.y;
              ptN.z=pt.z;
              if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                  Coordinates3D<double> hexCoordsP1=hexagonVertices[4]+hexCoords;
                  Coordinates3D<double> hexCoordsP2=hexagonVertices[5]+hexCoords;
                  local_points.push_back(std::pair<double, double>(hexCoordsP1.x, hexCoordsP1.y));
                  local_points.push_back(std::pair<double, double>(hexCoordsP2.x, hexCoordsP2.y));
              }
          }
          if(pt.x-1>=0 && pt.y+1<dim[1]){
              ptN.x=pt.x-1;
              ptN.y=pt.y+1;

              ptN.z=pt.z;
              if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                  Coordinates3D<double> hexCoordsP1=hexagonVertices[5]+hexCoords;
                  Coordinates3D<double> hexCoordsP2=hexagonVertices[0]+hexCoords;
                  local_points.push_back(std::pair<double, double>(hexCoordsP1.x, hexCoordsP1.y));
                  local_points.push_back(std::pair<double, double>(hexCoordsP2.x, hexCoordsP2.y));
              }
          }
          if( pt.y+1<dim[1]){
              ptN.x=pt.x;
              ptN.y=pt.y+1;
              ptN.z=pt.z;
              if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                  Coordinates3D<double> hexCoordsP1=hexagonVertices[0]+hexCoords;
                  Coordinates3D<double> hexCoordsP2=hexagonVertices[1]+hexCoords;
                  local_points.push_back(std::pair<double, double>(hexCoordsP1.x, hexCoordsP1.y));
                  local_points.push_back(std::pair<double, double>(hexCoordsP2.x, hexCoordsP2.y));
              }
          }
          if( pt.x+1<dim[0]){
              ptN.x=pt.x+1;
              ptN.y=pt.y;
              ptN.z=pt.z;
              if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                  Coordinates3D<double> hexCoordsP1=hexagonVertices[1]+hexCoords;
                  Coordinates3D<double> hexCoordsP2=hexagonVertices[2]+hexCoords;
                  local_points.push_back(std::pair<double, double>(hexCoordsP1.x, hexCoordsP1.y));
                  local_points.push_back(std::pair<double, double>(hexCoordsP2.x, hexCoordsP2.y));
              }
          }
          if( pt.y-1>=0){
              ptN.x=pt.x;
              ptN.y=pt.y-1;
              ptN.z=pt.z;
              if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                  Coordinates3D<double> hexCoordsP1=hexagonVertices[2]+hexCoords;
                  Coordinates3D<double> hexCoordsP2=hexagonVertices[3]+hexCoords;
                  local_points.push_back(std::pair<double, double>(hexCoordsP1.x, hexCoordsP1.y));
                  local_points.push_back(std::pair<double, double>(hexCoordsP2.x, hexCoordsP2.y));
              }
          }
          if(pt.x-1>=0 && pt.y-1>=0){
                ptN.x=pt.x-1;
                ptN.y=pt.y-1;
                ptN.z=pt.z;
                if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                    Coordinates3D<double> hexCoordsP1=hexagonVertices[3]+hexCoords;
                    Coordinates3D<double> hexCoordsP2=hexagonVertices[4]+hexCoords;
                    local_points.push_back(std::pair<double,double>(hexCoordsP1.x, hexCoordsP1.y));
                    local_points.push_back(std::pair<double,double>(hexCoordsP2.x, hexCoordsP2.y));
                }
            }
        }
      }
		}
  }
#pragma omp critical
  {
    global_points.insert(global_points.end(), local_points.begin(), local_points.end());
  }

#pragma omp barrier

#pragma omp sections
  {
#pragma omp section
    {
      linesWritePtr = lines->WritePointer(global_points.size() / 2, (global_points.size() / 2) * 3);
    }
#pragma omp section
    {
      points->SetNumberOfPoints(global_points.size());
    }
  }
#pragma omp for schedule(static)
  for (int j = 0; j < global_points.size(); j += 2)
  {
    std::pair<double, double> pt1 = global_points[j];
    std::pair<double, double> pt2 = global_points[j + 1];
    points->SetPoint(j, pt1.first, pt1.second, 0);
    points->SetPoint(j + 1, pt2.first, pt2.second, 0);
    int pc = (j / 2) * 3;
    linesWritePtr[pc] = 2;
    linesWritePtr[pc + 1] = j;
    linesWritePtr[pc + 2] = j + 1;
  }
}
pUtils->setNumberOfWorkNodesAuto(1);
}

void FieldExtractor::fillCentroidData2D(vtk_obj_addr_int_t _pointArrayAddr ,vtk_obj_addr_int_t _linesArrayAddr, std::string _plane ,  int _pos){
	CellInventory *cellInventoryPtr = &potts->getCellInventory();
	CellInventory::cellInventoryIterator cInvItr;
	CellG * cell;

	float x,y,z;

	vtkPoints *points = (vtkPoints *)_pointArrayAddr;
	vtkCellArray * lines=(vtkCellArray *)_linesArrayAddr;

	int ptCount=0;
	for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr ){
		cell = cellInventoryPtr->getCell(cInvItr);
		float cellVol = (float)cell->volume;
		if (!cell->volume) {
          exit(-1);
		}
		float xmid = (float)cell->xCM / cell->volume;
		float ymid = (float)cell->yCM / cell->volume;
		float R = sqrt((float)cell->volume) / 2.0;
        float x0 = xmid-R;
		float x1 = xmid+R;
		float y0 = ymid-R;
		float y1 = ymid+R;
		points->InsertNextPoint(x0,y0,0);
		points->InsertNextPoint(x1,y0,0);
		points->InsertNextPoint(x1,y1,0);
		points->InsertNextPoint(x0,y1,0);

		lines->InsertNextCell(5);
		lines->InsertCellPoint(ptCount);  ptCount++;
		lines->InsertCellPoint(ptCount);  ptCount++;
		lines->InsertCellPoint(ptCount);  ptCount++;
		lines->InsertCellPoint(ptCount);  ptCount++;
		lines->InsertCellPoint(ptCount-4);
	}
}

bool FieldExtractor::fillConFieldData2DHex(vtk_obj_addr_int_t _conArrayAddr,vtk_obj_addr_int_t _hexCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos)
{
  vtkDoubleArray *conArray=(vtkDoubleArray *)_conArrayAddr;
  vtkCellArray * _hexCellsArray=(vtkCellArray*)_hexCellsArrayAddr;
	vtkPoints *_pointsArray=(vtkPoints *)_pointsArrayAddr;

	Field3D<float> *conFieldPtr=0;
	std::map<std::string,Field3D<float>*> & fieldMap=sim->getConcentrationFieldNameMap();
	std::map<std::string,Field3D<float>*>::iterator mitr;
	mitr=fieldMap.find(_conFieldName);
	if(mitr!=fieldMap.end()){
		conFieldPtr=mitr->second;
	}

	if(!conFieldPtr)
		return false;

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

	vector<int> fieldDimVec(3,0);
	fieldDimVec[0]=fieldDim.x;
	fieldDimVec[1]=fieldDim.y;
	fieldDimVec[2]=fieldDim.z;

	vector<int> pointOrderVec=pointOrder(_plane);
	vector<int> dimOrderVec=dimOrder(_plane);

	vector<int> dim(3,0);
	dim[0]=fieldDimVec[dimOrderVec[0]];
	dim[1]=fieldDimVec[dimOrderVec[1]];
	dim[2]=fieldDimVec[dimOrderVec[2]];

  int numPoints = dim[0] * dim[1];
  vtkIdType *_hexWritePtr;
  pUtils->setNumberOfWorkNodesAuto();

#pragma omp parallel shared(pointOrderVec, dim, cellFieldG, _pointsArray, conArray, _hexWritePtr)
  {
#pragma omp sections
  {
#pragma omp section
    {
      _hexWritePtr = _hexCellsArray->WritePointer(numPoints, numPoints * 7);
    }
#pragma omp section
    {
      conArray->SetNumberOfValues(numPoints);
    }
#pragma omp section
    {
      _pointsArray->SetNumberOfPoints(numPoints * 6);
    }
  }
  Point3D pt;
	vector<int> ptVec(3,0);

	double con;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
#pragma omp for schedule(static)
  for(int j =0 ; j<dim[1] ; ++j) {
		for(int i =0 ; i<dim[0] ; ++i) {
      int dataPoint = i + j * dim[1];
      ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

			if (i==dim[0] || j==dim[1]){
				con=0.0;
			}else{
				con = conFieldPtr->get(pt);
			}
			Coordinates3D<double> hexCoords=HexCoordXY(pt.x,pt.y,pt.z);
      int cellPos = dataPoint * 6;
      for (int idx=0 ; idx<6 ; ++idx){
        Coordinates3D<double> hexagonVertex=hexagonVertices[idx]+hexCoords;
        _pointsArray->SetPoint(cellPos + idx, hexagonVertex.x, hexagonVertex.y, 0.0);
      }
      int arrPos = dataPoint * 7;
      _hexWritePtr[arrPos + 0] = 6;
      _hexWritePtr[arrPos + 1] = cellPos + 0;
      _hexWritePtr[arrPos + 2] = cellPos + 1;
      _hexWritePtr[arrPos + 3] = cellPos + 2;
      _hexWritePtr[arrPos + 4] = cellPos + 3;
      _hexWritePtr[arrPos + 5] = cellPos + 4;
      _hexWritePtr[arrPos + 6] = cellPos + 5;

      conArray->SetValue(dataPoint, con);
    }
  }
}
  pUtils->setNumberOfWorkNodesAuto(1);
return true;
}

bool FieldExtractor::fillConFieldData2DCartesian(vtk_obj_addr_int_t _conArrayAddr,vtk_obj_addr_int_t _cartesianCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){
  vtkDoubleArray *conArray=(vtkDoubleArray *)_conArrayAddr;
	vtkCellArray * _cartesianCellsArray=(vtkCellArray*)_cartesianCellsArrayAddr;
	vtkPoints *_pointsArray=(vtkPoints *)_pointsArrayAddr;

  Field3D<float> *conFieldPtr=0;
	std::map<std::string,Field3D<float>*> & fieldMap=sim->getConcentrationFieldNameMap();
	std::map<std::string,Field3D<float>*>::iterator mitr;
	mitr=fieldMap.find(_conFieldName);
	if(mitr!=fieldMap.end()){
		conFieldPtr=mitr->second;
	}

	if(!conFieldPtr)
		return false;

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

	vector<int> fieldDimVec(3,0);
	fieldDimVec[0]=fieldDim.x;
	fieldDimVec[1]=fieldDim.y;
	fieldDimVec[2]=fieldDim.z;

	vector<int> pointOrderVec=pointOrder(_plane);
	vector<int> dimOrderVec=dimOrder(_plane);

	vector<int> dim(3,0);
	dim[0]=fieldDimVec[dimOrderVec[0]];
	dim[1]=fieldDimVec[dimOrderVec[1]];
	dim[2]=fieldDimVec[dimOrderVec[2]];

  int numPoints = dim[0] * dim[1];
  vtkIdType *_cartesianCellsArrayWritePtr;
  pUtils->setNumberOfWorkNodesAuto();
#pragma omp parallel shared(pointOrderVec, dim, cellFieldG, _pointsArray, conArray, _cartesianCellsArrayWritePtr)
{
  #pragma omp sections
  {
    #pragma omp section
    {
      _cartesianCellsArrayWritePtr = _cartesianCellsArray->WritePointer(numPoints, numPoints * 5);
    }
#pragma omp section
    {
      conArray->SetNumberOfValues(numPoints);
    }
#pragma omp section
    {
      _pointsArray->SetNumberOfPoints(numPoints * 4);
    }
  }
  Point3D pt;
  vector<int> ptVec(3, 0);
  double con;
#pragma omp for schedule(static)
  for(int j =0 ; j<dim[1] ; ++j) {

    for(int i =0 ; i<dim[0] ; ++i){
      int dataPoint = i + j * dim[1];
      ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

			if (i==dim[0] || j==dim[1]){
				con=0.0;
			}else{
				con = conFieldPtr->get(pt);
			}

      Coordinates3D<double> coords(ptVec[0],ptVec[1],0); // notice that we are drawing pixels from other planes on a xy plan so we use ptVec instead of pt. pt is absolute position of the point ptVec is for projection purposes
      int cellPos = dataPoint * 4;
      for (int idx=0 ; idx<4 ; ++idx){
			  Coordinates3D<double> cartesianVertex=cartesianVertices[idx]+coords;
        _pointsArray->SetPoint(cellPos + idx, cartesianVertex.x, cartesianVertex.y, 0.0);
      }

      int arrPos = dataPoint * 5;
      _cartesianCellsArrayWritePtr[arrPos + 0] = 4;
      _cartesianCellsArrayWritePtr[arrPos + 1] = cellPos + 0;
      _cartesianCellsArrayWritePtr[arrPos + 2] = cellPos + 1;
      _cartesianCellsArrayWritePtr[arrPos + 3] = cellPos + 2;
      _cartesianCellsArrayWritePtr[arrPos + 4] = cellPos + 3;

      conArray->SetValue(dataPoint, con);
		}
  }
}
pUtils->setNumberOfWorkNodesAuto(1);
  return true;
}


bool FieldExtractor::fillScalarFieldData2DHex(vtk_obj_addr_int_t _conArrayAddr,vtk_obj_addr_int_t _hexCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){
  vtkDoubleArray *conArray=(vtkDoubleArray *)_conArrayAddr;
	vtkCellArray * _hexCellsArray=(vtkCellArray*)_hexCellsArrayAddr;
	vtkPoints *_pointsArray=(vtkPoints *)_pointsArrayAddr;

	FieldStorage::floatField3D_t * conFieldPtr=fsPtr->getScalarFieldByName(_conFieldName);

	if(!conFieldPtr)
		return false;

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

	vector<int> fieldDimVec(3,0);
	fieldDimVec[0]=fieldDim.x;
	fieldDimVec[1]=fieldDim.y;
	fieldDimVec[2]=fieldDim.z;

	vector<int> pointOrderVec=pointOrder(_plane);
	vector<int> dimOrderVec=dimOrder(_plane);

	vector<int> dim(3,0);
	dim[0]=fieldDimVec[dimOrderVec[0]];
	dim[1]=fieldDimVec[dimOrderVec[1]];
	dim[2]=fieldDimVec[dimOrderVec[2]];
  vtkIdType *_hexWritePtr;
  int numPoints = dim[0] * dim[1];
  pUtils->setNumberOfWorkNodesAuto();
#pragma omp parallel shared(pointOrderVec, dim, cellFieldG, _pointsArray, conArray, _hexWritePtr)
{
#pragma omp sections
  {
#pragma omp section
    {
      _hexWritePtr = _hexCellsArray->WritePointer(numPoints, numPoints * 7);
    }
#pragma omp section
    {
      conArray->SetNumberOfValues(numPoints);
    }
#pragma omp section
    {
      _pointsArray->SetNumberOfPoints(numPoints * 6);
    }
  }

	Point3D pt;
	vector<int> ptVec(3,0);
	double con;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
#pragma omp for schedule(static)
	for(int j =0 ; j<dim[1] ; ++j) {
		for(int i =0 ; i<dim[0] ; ++i){
      int dataPoint = i + j * dim[1];
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

			if (i==dim[0] || j==dim[1]){
				con=0.0;
			}else{
				con = (*conFieldPtr)[pt.x][pt.y][pt.z];
			}

			Coordinates3D<double> hexCoords=HexCoordXY(pt.x,pt.y,pt.z);

      int cellPos = dataPoint * 6;
			for (int idx=0 ; idx<6 ; ++idx){
		 	  Coordinates3D<double> hexagonVertex=hexagonVertices[idx]+hexCoords;
        _pointsArray->SetPoint(cellPos + idx, hexagonVertex.x, hexagonVertex.y, 0.0);
			}
      int arrPos = dataPoint * 7;
      _hexWritePtr[arrPos + 0] = 6;
      _hexWritePtr[arrPos + 1] = cellPos + 0;
      _hexWritePtr[arrPos + 2] = cellPos + 1;
      _hexWritePtr[arrPos + 3] = cellPos + 2;
      _hexWritePtr[arrPos + 4] = cellPos + 3;
      _hexWritePtr[arrPos + 5] = cellPos + 4;
      _hexWritePtr[arrPos + 6] = cellPos + 5;

      conArray->SetValue(dataPoint, con);
		}
  }
}
  pUtils->setNumberOfWorkNodesAuto();
  return true;
}

bool FieldExtractor::fillScalarFieldData2DCartesian(vtk_obj_addr_int_t _conArrayAddr,vtk_obj_addr_int_t _cartesianCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){
  vtkDoubleArray *conArray=(vtkDoubleArray *)_conArrayAddr;
	vtkCellArray * _cartesianCellsArray=(vtkCellArray*)_cartesianCellsArrayAddr;
	vtkPoints *_pointsArray=(vtkPoints *)_pointsArrayAddr;

	FieldStorage::floatField3D_t * conFieldPtr=fsPtr->getScalarFieldByName(_conFieldName);


	if(!conFieldPtr)
		return false;


	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

	vector<int> fieldDimVec(3,0);
	fieldDimVec[0]=fieldDim.x;
	fieldDimVec[1]=fieldDim.y;
	fieldDimVec[2]=fieldDim.z;

	vector<int> pointOrderVec=pointOrder(_plane);
	vector<int> dimOrderVec=dimOrder(_plane);

	vector<int> dim(3,0);
	dim[0]=fieldDimVec[dimOrderVec[0]];
	dim[1]=fieldDimVec[dimOrderVec[1]];
	dim[2]=fieldDimVec[dimOrderVec[2]];

  int numPoints = dim[0] * dim[1];
  vtkIdType *_cartesianCellsArrayWritePtr;
  pUtils->setNumberOfWorkNodesAuto();
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
#pragma omp parallel shared(pointOrderVec, dim, cellFieldG, _pointsArray, conArray, _cartesianCellsArrayWritePtr)
{

#pragma omp sections
{
#pragma omp section
{
  _cartesianCellsArrayWritePtr = _cartesianCellsArray->WritePointer(numPoints, numPoints * 5);
}
#pragma omp section
{
  conArray->SetNumberOfValues(numPoints);
}
#pragma omp section
{
  _pointsArray->SetNumberOfPoints(numPoints * 4);
}
}

#pragma omp for schedule(static)
  for(int j =0 ; j<dim[1] ; ++j) {
    Point3D pt;
    vector<int> ptVec(3, 0);

    double con;
    for(int i =0 ; i<dim[0] ; ++i){
      int dataPoint = i + j * dim[1];
      ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

			if (i==dim[0] || j==dim[1]){
				con=0.0;
			}else{
				con = (*conFieldPtr)[pt.x][pt.y][pt.z];
			}

      Coordinates3D<double> coords(ptVec[0],ptVec[1],0); // notice that we are drawing pixels from other planes on a xy plan so we use ptVec instead of pt. pt is absolute position of the point ptVec is for projection purposes
      int cellPos = dataPoint * 4;
      for (int idx=0 ; idx<4 ; ++idx){
			  Coordinates3D<double> cartesianVertex=cartesianVertices[idx]+coords;
        _pointsArray->SetPoint(cellPos + idx, cartesianVertex.x, cartesianVertex.y, 0.0);
      }

      int arrPos = dataPoint * 5;
      _cartesianCellsArrayWritePtr[arrPos + 0] = 4;
      _cartesianCellsArrayWritePtr[arrPos + 1] = cellPos + 0;
      _cartesianCellsArrayWritePtr[arrPos + 2] = cellPos + 1;
      _cartesianCellsArrayWritePtr[arrPos + 3] = cellPos + 2;
      _cartesianCellsArrayWritePtr[arrPos + 4] = cellPos + 3;

      conArray->SetValue(dataPoint, con);
		}
  }
}
  pUtils->setNumberOfWorkNodesAuto(1);
  return true;
}

bool FieldExtractor::fillScalarFieldCellLevelData2DHex(vtk_obj_addr_int_t _conArrayAddr,vtk_obj_addr_int_t _hexCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){
	vtkDoubleArray *conArray=(vtkDoubleArray *)_conArrayAddr;
	vtkCellArray * _hexCellsArray=(vtkCellArray*)_hexCellsArrayAddr;
	vtkPoints *_pointsArray=(vtkPoints *)_pointsArrayAddr;

	FieldStorage::scalarFieldCellLevel_t * conFieldPtr=fsPtr->getScalarFieldCellLevelFieldByName(_conFieldName);

	if(!conFieldPtr)
		return false;

	FieldStorage::scalarFieldCellLevel_t::iterator mitr;

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

	vector<int> fieldDimVec(3,0);
	fieldDimVec[0]=fieldDim.x;
	fieldDimVec[1]=fieldDim.y;
	fieldDimVec[2]=fieldDim.z;

	vector<int> pointOrderVec=pointOrder(_plane);
	vector<int> dimOrderVec=dimOrder(_plane);

	vector<int> dim(3,0);
	dim[0]=fieldDimVec[dimOrderVec[0]];
	dim[1]=fieldDimVec[dimOrderVec[1]];
	dim[2]=fieldDimVec[dimOrderVec[2]];
  int numPoints = dim[0] * dim[1];
  vtkIdType *_hexWritePtr;

  pUtils->setNumberOfWorkNodesAuto();
#pragma omp parallel shared(pointOrderVec, dim, cellFieldG, _pointsArray, conArray, _hexWritePtr)
{
#pragma omp sections
  {
#pragma omp section
    {
      _hexWritePtr = _hexCellsArray->WritePointer(numPoints, numPoints * 7);
    }
#pragma omp section
    {
      conArray->SetNumberOfValues(numPoints);
    }
#pragma omp section
    {
      _pointsArray->SetNumberOfPoints(numPoints * 6);
    }
  }

	Point3D pt;
	vector<int> ptVec(3,0);
	CellG *cell;
	double con;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
	for(int j =0 ; j<dim[1] ; ++j) {
		for(int i =0 ; i<dim[0] ; ++i){
      int dataPoint = i + j * dim[1];
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

			cell=cellFieldG->get(pt);
			if (i==dim[0] || j==dim[1]){
				con=0.0;
			}else{
				if(cell){
					mitr=conFieldPtr->find(cell);
					if(mitr!=conFieldPtr->end()){
						con=mitr->second;
					}else{
						con=0.0;
					}
				}else{
					con=0.0;
				}
			}
			Coordinates3D<double> hexCoords=HexCoordXY(pt.x,pt.y,pt.z);
      int cellPos = dataPoint * 6;
			for (int idx=0 ; idx<6 ; ++idx){
			  Coordinates3D<double> hexagonVertex=hexagonVertices[idx]+hexCoords;
 			  _pointsArray->SetPoint(cellPos + idx, hexagonVertex.x, hexagonVertex.y, 0.0);
			}
      int arrPos = dataPoint * 7;
      _hexWritePtr[arrPos + 0] = 6;
      _hexWritePtr[arrPos + 1] = cellPos + 0;
      _hexWritePtr[arrPos + 2] = cellPos + 1;
      _hexWritePtr[arrPos + 3] = cellPos + 2;
      _hexWritePtr[arrPos + 4] = cellPos + 3;
      _hexWritePtr[arrPos + 5] = cellPos + 4;
      _hexWritePtr[arrPos + 6] = cellPos + 5;

      conArray->SetValue(dataPoint, con);
		}
  }
}
  pUtils->setNumberOfWorkNodesAuto(1);
  return true;
}


bool FieldExtractor::fillScalarFieldCellLevelData2DCartesian(vtk_obj_addr_int_t _conArrayAddr,vtk_obj_addr_int_t _cartesianCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){
	vtkDoubleArray *conArray=(vtkDoubleArray *)_conArrayAddr;
	vtkCellArray * _cartesianCellsArray=(vtkCellArray*)_cartesianCellsArrayAddr;
	vtkPoints *_pointsArray=(vtkPoints *)_pointsArrayAddr;

	FieldStorage::scalarFieldCellLevel_t * conFieldPtr=fsPtr->getScalarFieldCellLevelFieldByName(_conFieldName);

	if(!conFieldPtr)
		return false;

	FieldStorage::scalarFieldCellLevel_t::iterator mitr;

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

	vector<int> fieldDimVec(3,0);
	fieldDimVec[0]=fieldDim.x;
	fieldDimVec[1]=fieldDim.y;
	fieldDimVec[2]=fieldDim.z;

	vector<int> pointOrderVec=pointOrder(_plane);
	vector<int> dimOrderVec=dimOrder(_plane);

	vector<int> dim(3,0);
	dim[0]=fieldDimVec[dimOrderVec[0]];
	dim[1]=fieldDimVec[dimOrderVec[1]];
	dim[2]=fieldDimVec[dimOrderVec[2]];
  int numPoints = dim[0] * dim[1];
  vtkIdType *_cartesianCellsArrayWritePtr;
  pUtils->setNumberOfWorkNodesAuto();
#pragma omp parallel shared(pointOrderVec, dim, cellFieldG, _pointsArray, conArray, _cartesianCellsArrayWritePtr)
{
  #pragma omp sections
  {
    #pragma omp section
    {
      _cartesianCellsArrayWritePtr = _cartesianCellsArray->WritePointer(numPoints, numPoints * 5);
    }
#pragma omp section
    {
      conArray->SetNumberOfValues(numPoints);
    }
#pragma omp section
    {
      _pointsArray->SetNumberOfPoints(numPoints * 4);
    }
  }
	Point3D pt;
	vector<int> ptVec(3,0);
	CellG *cell;
	double con;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned

#pragma omp for schedule(static)
	for(int j =0 ; j<dim[1] ; ++j) {
		for(int i =0 ; i<dim[0] ; ++i){
      int dataPoint = i + j * dim[1];
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

			cell=cellFieldG->get(pt);
			if (i==dim[0] || j==dim[1]){
				con=0.0;
			}else{
				if(cell){
					mitr=conFieldPtr->find(cell);
					if(mitr!=conFieldPtr->end()){
						con=mitr->second;
					}else{
						con=0.0;
					}
				}else{
					con=0.0;
				}
			}

      Coordinates3D<double> coords(ptVec[0],ptVec[1],0); // notice that we are drawing pixels from other planes on a xy plan so we use ptVec instead of pt. pt is absolute position of the point ptVec is for projection purposes
      int cellPos = dataPoint * 4;
			for (int idx=0 ; idx<4 ; ++idx){
			  Coordinates3D<double> cartesianVertex=cartesianVertices[idx]+coords;
 			  _pointsArray->SetPoint(cellPos + idx, cartesianVertex.x, cartesianVertex.y, 0.0);
			}

      int arrPos = dataPoint * 5;
      _cartesianCellsArrayWritePtr[arrPos + 0] = 4;
      _cartesianCellsArrayWritePtr[arrPos + 1] = cellPos + 0;
      _cartesianCellsArrayWritePtr[arrPos + 2] = cellPos + 1;
      _cartesianCellsArrayWritePtr[arrPos + 3] = cellPos + 2;
      _cartesianCellsArrayWritePtr[arrPos + 4] = cellPos + 3;

      conArray->SetValue(dataPoint, con);
		}
  }
}
  pUtils->setNumberOfWorkNodesAuto(1);
  return true;
}


bool FieldExtractor::fillConFieldData2D(vtk_obj_addr_int_t _conArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos){
  vtkDoubleArray *conArray=(vtkDoubleArray *)_conArrayAddr;
	Field3D<float> *conFieldPtr=0;
	std::map<std::string,Field3D<float>*> & fieldMap=sim->getConcentrationFieldNameMap();
	std::map<std::string,Field3D<float>*>::iterator mitr;
	mitr=fieldMap.find(_conFieldName);
	if(mitr!=fieldMap.end()){
		conFieldPtr=mitr->second;
	}

	if(!conFieldPtr)
		return false;


	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

	vector<int> fieldDimVec(3,0);
	fieldDimVec[0]=fieldDim.x;
	fieldDimVec[1]=fieldDim.y;
	fieldDimVec[2]=fieldDim.z;

	vector<int> pointOrderVec=pointOrder(_plane);
	vector<int> dimOrderVec=dimOrder(_plane);

	vector<int> dim(3,0);
	dim[0]=fieldDimVec[dimOrderVec[0]];
	dim[1]=fieldDimVec[dimOrderVec[1]];
	dim[2]=fieldDimVec[dimOrderVec[2]];


	conArray->SetNumberOfValues((dim[1]+2)*(dim[0]+1));
	//For some reasons the points x=0 are eaten up (don't know why).
	//So we just populate concentration 0.0.
    pUtils->setNumberOfWorkNodesAuto();
#pragma omp parallel shared(pointOrderVec, dim, conFieldPtr, conArray)
  {
#pragma omp for nowait schedule(static)
  for (int i = 0 ; i< dim[0]+1 ;++i){
		conArray->SetValue(i, 0.0);
	}

	Point3D pt;
  vector<int> ptVec(3, 0);
  double con;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
#pragma omp for nowait schedule(static)
  for(int j =0 ; j<dim[1]+1 ; ++j)
		for(int i =0 ; i<dim[0]+1 ; ++i){
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

			if (i==dim[0] || j==dim[1]){
				con=0.0;
			}else{
				con = conFieldPtr->get(pt);
			}
      int pos = i + j * (dim[1] + 1) + (dim[0] + 1);
      conArray->SetValue(pos, con);
		}
}
  pUtils->setNumberOfWorkNodesAuto(1);
return true;
}

bool FieldExtractor::fillScalarFieldData2D(vtk_obj_addr_int_t _conArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos){
  vtkDoubleArray *conArray=(vtkDoubleArray *)_conArrayAddr;
	FieldStorage::floatField3D_t * conFieldPtr=fsPtr->getScalarFieldByName(_conFieldName);

	if(!conFieldPtr)
		return false;

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

	vector<int> fieldDimVec(3,0);
	fieldDimVec[0]=fieldDim.x;
	fieldDimVec[1]=fieldDim.y;
	fieldDimVec[2]=fieldDim.z;

	vector<int> pointOrderVec=pointOrder(_plane);
	vector<int> dimOrderVec=dimOrder(_plane);

	vector<int> dim(3,0);
	dim[0]=fieldDimVec[dimOrderVec[0]];
	dim[1]=fieldDimVec[dimOrderVec[1]];
	dim[2]=fieldDimVec[dimOrderVec[2]];

	conArray->SetNumberOfValues((dim[1]+2)*(dim[0]+1));
	//For some reasons the points x=0 are eaten up (don't know why).
	//So we just populate concentration 0.0.
    pUtils->setNumberOfWorkNodesAuto();
#pragma omp parallel shared(pointOrderVec, dim, conFieldPtr, conArray)
{
  #pragma omp for nowait schedule(static)
	for (int i = 0 ; i< dim[0]+1 ;++i){
		conArray->SetValue(i, 0.0);

	}

	Point3D pt;
	vector<int> ptVec(3,0);

	double con;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
#pragma omp for nowait schedule(static)
	for(int j =0 ; j<dim[1]+1 ; ++j){
		for(int i =0 ; i<dim[0]+1 ; ++i){
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

			if (i==dim[0] || j==dim[1]){
				con=0.0;
			}else{
				con = (*conFieldPtr)[pt.x][pt.y][pt.z];
			}
      int pos = i + j * (dim[1] + 1) + (dim[0] + 1);
			conArray->SetValue(pos, con);
		}
  }
}
  pUtils->setNumberOfWorkNodesAuto(1);
  return true;
}

bool FieldExtractor::fillScalarFieldCellLevelData2D(vtk_obj_addr_int_t _conArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos){
  vtkDoubleArray *conArray=(vtkDoubleArray *)_conArrayAddr;
	FieldStorage::scalarFieldCellLevel_t * conFieldPtr=fsPtr->getScalarFieldCellLevelFieldByName(_conFieldName);

	if(!conFieldPtr)
		return false;

	FieldStorage::scalarFieldCellLevel_t::iterator mitr;

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

	vector<int> fieldDimVec(3,0);
	fieldDimVec[0]=fieldDim.x;
	fieldDimVec[1]=fieldDim.y;
	fieldDimVec[2]=fieldDim.z;

	vector<int> pointOrderVec=pointOrder(_plane);
	vector<int> dimOrderVec=dimOrder(_plane);

	vector<int> dim(3,0);
	dim[0]=fieldDimVec[dimOrderVec[0]];
	dim[1]=fieldDimVec[dimOrderVec[1]];
	dim[2]=fieldDimVec[dimOrderVec[2]];

	conArray->SetNumberOfValues((dim[1]+2)*(dim[0]+1));
	//For some reasons the points x=0 are eaten up (don't know why).
	//So we just populate concentration 0.0.
    pUtils->setNumberOfWorkNodesAuto();
#pragma omp parallel shared(pointOrderVec, dim, conFieldPtr, conArray)
{
  #pragma omp for nowait schedule(static)
	for (int i = 0 ; i< dim[0]+1 ;++i){
		conArray->SetValue(i, 0.0);
	}

	Point3D pt;
	vector<int> ptVec(3,0);
	CellG* cell;

	double con;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
#pragma omp for nowait schedule(static)
	for(int j =0 ; j<dim[1]+1 ; ++j) {
		for(int i =0 ; i<dim[0]+1 ; ++i){
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

			cell=cellFieldG->get(pt);
			if (i==dim[0] || j==dim[1]){
				con=0.0;
			}else{

				if(cell){
					mitr=conFieldPtr->find(cell);
					if(mitr!=conFieldPtr->end()){
						con=mitr->second;
					}else{
						con=0.0;
					}
				}else{
					con=0.0;
				}
			}
      int pos = i + j * (dim[1] + 1) + (dim[0] + 1);
      conArray->SetValue(pos, con);
		}
  }
}
 pUtils->setNumberOfWorkNodesAuto(1);
  return true;
}

bool FieldExtractor::fillVectorFieldData2D(vtk_obj_addr_int_t _pointsArrayIntAddr,vtk_obj_addr_int_t _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos){
  vtkFloatArray * vectorArray=(vtkFloatArray *)_vectorArrayIntAddr;
	vtkPoints *pointsArray=(vtkPoints *)_pointsArrayIntAddr;

	FieldStorage::vectorField3D_t * vectorFieldPtr=fsPtr->getVectorFieldFieldByName(_fieldName);

	if(!vectorFieldPtr)
		return false;

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

	vector<int> fieldDimVec(3,0);
	fieldDimVec[0]=fieldDim.x;
	fieldDimVec[1]=fieldDim.y;
	fieldDimVec[2]=fieldDim.z;

	vector<int> pointOrderVec=pointOrder(_plane);
	vector<int> dimOrderVec=dimOrder(_plane);

	vector<int> dim(3,0);
	dim[0]=fieldDimVec[dimOrderVec[0]];
	dim[1]=fieldDimVec[dimOrderVec[1]];
	dim[2]=fieldDimVec[dimOrderVec[2]];
  vector<std::tuple<short,short,float,float>> globalPoints;

  pUtils->setNumberOfWorkNodesAuto(0);
#pragma omp parallel shared(pointOrderVec, dim, cellFieldG, pointsArray, vectorArray)
{
	Point3D pt;
	vector<int> ptVec(3,0);
	float  vecTmpCoord[3];
  float x,y,z;
	double con;
  vector<std::tuple<short,short,float,float>> localPoints;

  #pragma omp for nowait schedule(static)
	for(int j =0 ; j<dim[1] ; ++j) {
		for(int i =0 ; i<dim[0] ; ++i){
      int offset = i + j * dim[1];
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

      x=(*vectorFieldPtr)[pt.x][pt.y][pt.z][0];
      y=(*vectorFieldPtr)[pt.x][pt.y][pt.z][1];
      z=(*vectorFieldPtr)[pt.x][pt.y][pt.z][2];

      vecTmpCoord[0]=x;
      vecTmpCoord[1]=y;
      vecTmpCoord[2]=z;

			if(x!=0.0 || y!=0.0 || z!=0.0){
        localPoints.push_back(make_tuple(pt.x,pt.y, x,y));
			}
		}
  }
  #pragma omp critical
  {
    globalPoints.insert(globalPoints.end(), localPoints.begin(), localPoints.end());
  }
  #pragma omp barrier

#pragma omp sections
{
  #pragma omp section
  {
    pointsArray->SetNumberOfPoints(globalPoints.size());
  }
  #pragma omp section
  {
    vectorArray->SetNumberOfComponents(3);
    vectorArray->SetNumberOfTuples(globalPoints.size());
  }
}
#pragma omp for schedule(static)
  for (int i = 0; i < globalPoints.size(); ++i) {
    auto point = globalPoints[i];
    pointsArray->SetPoint(i, std::get<0>(point),std::get<1>(point),0.0);
    vectorArray->SetTuple3(i,std::get<2>(point),std::get<3>(point),0.0);
  }
}
  pUtils->setNumberOfWorkNodesAuto(1);
  return true;
}

bool FieldExtractor::fillVectorFieldData2DHex(vtk_obj_addr_int_t _pointsArrayIntAddr,vtk_obj_addr_int_t _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos){
  vtkFloatArray * vectorArray=(vtkFloatArray *)_vectorArrayIntAddr;
	vtkPoints *pointsArray=(vtkPoints *)_pointsArrayIntAddr;

	FieldStorage::vectorField3D_t * vectorFieldPtr=fsPtr->getVectorFieldFieldByName(_fieldName);

	if(!vectorFieldPtr)
		return false;

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

	vector<int> fieldDimVec(3,0);
	fieldDimVec[0]=fieldDim.x;
	fieldDimVec[1]=fieldDim.y;
	fieldDimVec[2]=fieldDim.z;

	vector<int> pointOrderVec=pointOrder(_plane);
	vector<int> dimOrderVec=dimOrder(_plane);

	vector<int> dim(3,0);
	dim[0]=fieldDimVec[dimOrderVec[0]];
	dim[1]=fieldDimVec[dimOrderVec[1]];
	dim[2]=fieldDimVec[dimOrderVec[2]];
  vector<std::tuple<double,double,float,float>> globalPoints;
  pUtils->setNumberOfWorkNodesAuto();
#pragma omp parallel shared(pointOrderVec, dim, vectorFieldPtr, pointsArray, vectorArray, globalPoints)
{
	Point3D pt;
	vector<int> ptVec(3,0);
	float  vecTmpCoord[3] ;
  float x,y,z;
  vector<std::tuple<double,double,float,float>> localPoints;
#pragma omp for nowait schedule(static)
	for(int j =0 ; j<dim[1] ; ++j) {
		for(int i =0 ; i<dim[0] ; ++i){
      int offset = i + j * dim[1];

			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

      x=(*vectorFieldPtr)[pt.x][pt.y][pt.z][0];
      y=(*vectorFieldPtr)[pt.x][pt.y][pt.z][1];
      z=(*vectorFieldPtr)[pt.x][pt.y][pt.z][2];

      vecTmpCoord[0]=x;
      vecTmpCoord[1]=y;
      vecTmpCoord[2]=z;

			if(x!=0.0 || y!=0.0 || z!=0.0){
				Coordinates3D<double> hexCoords=HexCoordXY(pt.x,pt.y,pt.z);
        localPoints.push_back(std::tuple<double,double,float,float>(hexCoords.x,hexCoords.y,vecTmpCoord[pointOrderVec[0]],vecTmpCoord[pointOrderVec[1]]));
			}
		}
  }
  #pragma omp critical
  {
    globalPoints.insert(globalPoints.end(), localPoints.begin(), localPoints.end());
  }

#pragma omp barrier

#pragma omp sections
{
  #pragma omp section
  {
    pointsArray->SetNumberOfPoints(globalPoints.size());
  }
  #pragma omp section
  {
    vectorArray->SetNumberOfComponents(3);
    vectorArray->SetNumberOfTuples(globalPoints.size());
  }
}

#pragma omp for schedule(static)
  for (int i = 0; i < globalPoints.size(); ++i)
  {
    auto point = globalPoints[i];
    pointsArray->SetPoint(i, std::get<0>(point),std::get<1>(point),0.0);
    vectorArray->SetTuple3(i,std::get<2>(point),std::get<3>(point),0.0);
  }
}
  pUtils->setNumberOfWorkNodesAuto(1);
  return true;
}

bool FieldExtractor::fillVectorFieldData3D(vtk_obj_addr_int_t _pointsArrayIntAddr,vtk_obj_addr_int_t _vectorArrayIntAddr,std::string _fieldName){
  vtkFloatArray * vectorArray=(vtkFloatArray *)_vectorArrayIntAddr;
	vtkPoints *pointsArray=(vtkPoints *)_pointsArrayIntAddr;

	FieldStorage::vectorField3D_t * vectorFieldPtr=fsPtr->getVectorFieldFieldByName(_fieldName);

	if(!vectorFieldPtr)
		return false;

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();
  vector<std::tuple<short,short,short, float,float,float>> globalPoints;
  pUtils->setNumberOfWorkNodesAuto();
#pragma omp parallel shared(vectorFieldPtr, pointsArray, vectorArray, globalPoints, fieldDim)
{
	Point3D pt;
  short pt_z;
  vector<std::tuple<short,short,short, float,float,float>> localPoints;
  float x,y,z;

// TODO: need OpenMP 3.0 > support on Windows to allow non-integer for loop indicies, cannot parallelize this
#pragma omp for nowait schedule(static)
	for(pt_z = 0; pt_z<fieldDim.z ; ++pt_z)	{
    pt.z=pt.z;
		for(pt.y =0 ; pt.y<fieldDim.y ; ++pt.y) {
			for(pt.x =0 ; pt.x<fieldDim.x ; ++pt.x) {
        x=(*vectorFieldPtr)[pt.x][pt.y][pt.z][0];
        y=(*vectorFieldPtr)[pt.x][pt.y][pt.z][1];
        z=(*vectorFieldPtr)[pt.x][pt.y][pt.z][2];
				if(x!=0.0 || y!=0.0 || z!=0.0){
          localPoints.push_back(make_tuple(pt.x,pt.y,pt.z, x,y,z));
				}
			}
    }
  }
  #pragma omp critical
  {
    globalPoints.insert(globalPoints.end(), localPoints.begin(), localPoints.end());
  }

#pragma omp barrier
#pragma omp sections
{
  #pragma omp section
  {
    pointsArray->SetNumberOfPoints(globalPoints.size());
  }
  #pragma omp section
  {
    vectorArray->SetNumberOfComponents(3);
    vectorArray->SetNumberOfTuples(globalPoints.size());
  }
}

#pragma omp for schedule(static)
  for (int i = 0; i < globalPoints.size(); ++i) {
    auto point = globalPoints[i];
    pointsArray->SetPoint(i, std::get<0>(point),std::get<1>(point),std::get<2>(point));
    vectorArray->SetTuple3(i,std::get<3>(point),std::get<4>(point),std::get<5>(point));
  }
}
    pUtils->setNumberOfWorkNodesAuto(1);
  return true;
}


bool FieldExtractor::fillVectorFieldData3DHex(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName) {
  vtkFloatArray *vectorArray = (vtkFloatArray *)_vectorArrayIntAddr;
  vtkPoints *pointsArray = (vtkPoints *)_pointsArrayIntAddr;

  FieldStorage::vectorField3D_t *vectorFieldPtr = fsPtr->getVectorFieldFieldByName(_fieldName);

  if (!vectorFieldPtr)
    return false;

  Field3D<CellG *> *cellFieldG = potts->getCellFieldG();
  Dim3D fieldDim = cellFieldG->getDim();
  vector<std::tuple<double,double,double, float,float,float>> globalPoints;
  pUtils->setNumberOfWorkNodesAuto();
#pragma omp parallel shared(vectorFieldPtr, pointsArray, vectorArray, globalPoints, fieldDim)
{
  Point3D pt;
  float x, y, z;
  int pt_z;
  vector<std::tuple<double,double,double, float,float,float>> localPoints;
// TODO: need OpenMP 3.0 > support on Windows to allow non-integer for loop indicies, cannot parallelize this
#pragma omp for nowait schedule(static)
  for (int pt_z = 0; pt_z < fieldDim.z; ++pt_z) {
    pt.z=pt.z;
    for (pt.y = 0; pt.y < fieldDim.y; ++pt.y) {
      for (pt.x = 0; pt.x < fieldDim.x; ++pt.x) {
        x = (*vectorFieldPtr)[pt.x][pt.y][pt.z][0];
        y = (*vectorFieldPtr)[pt.x][pt.y][pt.z][1];
        z = (*vectorFieldPtr)[pt.x][pt.y][pt.z][2];
        if (x != 0.0 || y != 0.0 || z != 0.0)
        {
          Coordinates3D<double> hexCoords = HexCoordXY(pt.x, pt.y, pt.z);
          localPoints.push_back(make_tuple(pt.x,pt.y,pt.z, x,y,z));
        }
      }
    }
  }
  #pragma omp critical
  {
    globalPoints.insert(globalPoints.end(), localPoints.begin(), localPoints.end());
  }

#pragma omp barrier
#pragma omp sections
{
  #pragma omp section
  {
    pointsArray->SetNumberOfPoints(globalPoints.size());
  }
  #pragma omp section
  {
    vectorArray->SetNumberOfComponents(3);
    vectorArray->SetNumberOfTuples(globalPoints.size());
  }
}

#pragma omp for schedule(static)
  for (int i = 0; i < globalPoints.size(); ++i) {
    auto point = globalPoints[i];
    pointsArray->SetPoint(i, std::get<0>(point),std::get<1>(point),std::get<2>(point));
    vectorArray->SetTuple3(i,std::get<3>(point),std::get<4>(point),std::get<5>(point));
  }
}
  pUtils->setNumberOfWorkNodesAuto(1);
  return true;
}


bool FieldExtractor::fillVectorFieldCellLevelData2D(vtk_obj_addr_int_t _pointsArrayIntAddr,vtk_obj_addr_int_t _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos){
  vtkFloatArray * vectorArray=(vtkFloatArray *)_vectorArrayIntAddr;
	vtkPoints *pointsArray=(vtkPoints *)_pointsArrayIntAddr;

	FieldStorage::vectorFieldCellLevel_t * vectorFieldPtr=fsPtr->getVectorFieldCellLevelFieldByName(_fieldName);

	if(!vectorFieldPtr)
		return false;

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

	vector<int> fieldDimVec(3,0);
	fieldDimVec[0]=fieldDim.x;
	fieldDimVec[1]=fieldDim.y;
	fieldDimVec[2]=fieldDim.z;

	vector<int> pointOrderVec=pointOrder(_plane);
	vector<int> dimOrderVec=dimOrder(_plane);

	vector<int> dim(3,0);
	dim[0]=fieldDimVec[dimOrderVec[0]];
	dim[1]=fieldDimVec[dimOrderVec[1]];
	dim[2]=fieldDimVec[dimOrderVec[2]];

  int numPoints = dim[1] * dim[0];
  vector<std::tuple<long,float,float,float,float>> globalPoints;
  set<long> globalVisitedCells;
  pUtils->setNumberOfWorkNodesAuto();
#pragma omp parallel shared(pointOrderVec, dim, vectorFieldPtr, pointsArray, vectorArray, globalVisitedCells, globalPoints)
{
	set<long> visitedCells;
	Point3D pt;
	vector<int> ptVec(3,0);
	CellG* cell;
	Coordinates3D<float> vecTmp;
	float vecTmpCoord[3];
  vector<std::tuple<long,float,float,float,float>> localPoints;

  #pragma omp for nowait schedule(static)
	for(int j =0 ; j<dim[1] ; ++j) {
		for(int i =0 ; i<dim[0] ; ++i){
      int dataPoint = i + j * dim[1];
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

			cell=cellFieldG->get(pt);

			if(cell){
				//check if this cell is in the set of visited Cells
				if(visitedCells.find(cell->id)!=visitedCells.end()){
					continue; //cell have been visited
				}else{
					//this is first time we visit given cell
					FieldStorage::vectorFieldCellLevelItr_t mitr=vectorFieldPtr->find(cell);
					if(mitr!=vectorFieldPtr->end()){
						vecTmp=mitr->second;
						vecTmpCoord[0]=vecTmp.x;
						vecTmpCoord[1]=vecTmp.y;
						vecTmpCoord[2]=vecTmp.z;
						localPoints.push_back(std::tuple<long,float,float,float,float>(cell->id, ptVec[0], ptVec[1], vecTmpCoord[pointOrderVec[0]],vecTmpCoord[pointOrderVec[1]]));
					}
					visitedCells.insert(cell->id);
				}
			}
		}
  }
  #pragma omp critical
  {
    for (auto item : localPoints) {
      if (globalVisitedCells.find(std::get<0>(item))==globalVisitedCells.end()) {
        globalPoints.push_back(item);
        globalVisitedCells.insert(std::get<0>(item));
      }
    }
  }

#pragma omp barrier

#pragma omp sections
{
  #pragma omp section
  {
    pointsArray->SetNumberOfPoints(globalPoints.size());
  }
  #pragma omp section
  {
    vectorArray->SetNumberOfComponents(3);
    vectorArray->SetNumberOfTuples(globalPoints.size());
  }
}

#pragma omp for schedule(static)
  for (int i=0; i < globalPoints.size(); ++i) {
    auto point = globalPoints[i];
    pointsArray->SetPoint(i, std::get<1>(point), std::get<2>(point), 0.0);
    vectorArray->SetTuple3(i,std::get<3>(point),std::get<4>(point),0.0);
  }
}
  pUtils->setNumberOfWorkNodesAuto(1);
  return true;
}

bool FieldExtractor::fillVectorFieldCellLevelData2DHex(vtk_obj_addr_int_t _pointsArrayIntAddr,vtk_obj_addr_int_t _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos){
  vtkFloatArray * vectorArray=(vtkFloatArray *)_vectorArrayIntAddr;
	vtkPoints *pointsArray=(vtkPoints *)_pointsArrayIntAddr;

	set<CellG*> visitedCells;

	FieldStorage::vectorFieldCellLevel_t * vectorFieldPtr=fsPtr->getVectorFieldCellLevelFieldByName(_fieldName);

	if(!vectorFieldPtr)
		return false;

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

	vector<int> fieldDimVec(3,0);
	fieldDimVec[0]=fieldDim.x;
	fieldDimVec[1]=fieldDim.y;
	fieldDimVec[2]=fieldDim.z;

	vector<int> pointOrderVec=pointOrder(_plane);
	vector<int> dimOrderVec=dimOrder(_plane);

	vector<int> dim(3,0);
	dim[0]=fieldDimVec[dimOrderVec[0]];
	dim[1]=fieldDimVec[dimOrderVec[1]];
	dim[2]=fieldDimVec[dimOrderVec[2]];
  vector<std::tuple<long,double,double,float,float>> globalPoints;
  set<long> globalVisitedCells;
  pUtils->setNumberOfWorkNodesAuto();
#pragma omp parallel shared(pointOrderVec, dim, vectorFieldPtr, pointsArray, vectorArray, globalVisitedCells, globalPoints)
{
	Point3D pt;
	vector<int> ptVec(3,0);
	CellG* cell;
	Coordinates3D<float> vecTmp;
	float vecTmpCoord[3];
  vector<std::tuple<long,double,double,float,float>> localPoints;
  set<long> visitedCells;

#pragma omp for schedule(static) nowait
	for(int j =0 ; j<dim[1] ; ++j) {
		for(int i =0 ; i<dim[0] ; ++i){
      int offset = i + j * dim[1];
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

			cell=cellFieldG->get(pt);

			if(cell){
				//check if this cell is in the set of visited Cells
				if(visitedCells.find(cell->id)!=visitedCells.end()){
					continue; //cell have been visited
				}else{
					//this is first time we visit given cell
					FieldStorage::vectorFieldCellLevelItr_t mitr=vectorFieldPtr->find(cell);
					if(mitr!=vectorFieldPtr->end()){
						vecTmp=mitr->second;
						vecTmpCoord[0]=vecTmp.x;
						vecTmpCoord[1]=vecTmp.y;
						vecTmpCoord[2]=vecTmp.z;
						Coordinates3D<double> hexCoords=HexCoordXY(pt.x,pt.y,pt.z);
						localPoints.push_back(make_tuple(cell->id, hexCoords.x,hexCoords.y, vecTmpCoord[pointOrderVec[0]],vecTmpCoord[pointOrderVec[1]]));
					}
					visitedCells.insert(cell->id);
				}
			}
		}
  }
#pragma omp critical
  {
    for (auto item : localPoints) {
      if (globalVisitedCells.find(std::get<0>(item))==globalVisitedCells.end()) {
        globalPoints.push_back(item);
        globalVisitedCells.insert(std::get<0>(item));
      }
    }
  }
#pragma omp barrier

#pragma omp sections
{
  #pragma omp section
  {
    pointsArray->SetNumberOfPoints(globalPoints.size());
  }
  #pragma omp section
  {
    vectorArray->SetNumberOfComponents(3);
    vectorArray->SetNumberOfTuples(globalPoints.size());
  }
}

#pragma omp for schedule(static)
  for (int i=0; i < globalPoints.size(); ++i) {
    auto point = globalPoints[i];
    pointsArray->SetPoint(i, std::get<1>(point), std::get<2>(point), 0.0);
    vectorArray->SetTuple3(i,std::get<3>(point),std::get<4>(point),0.0);
  }
}
  pUtils->setNumberOfWorkNodesAuto(1);
  return true;
}

bool FieldExtractor::fillVectorFieldCellLevelData3D(vtk_obj_addr_int_t _pointsArrayIntAddr,vtk_obj_addr_int_t _vectorArrayIntAddr,std::string _fieldName){
  vtkFloatArray * vectorArray=(vtkFloatArray *)_vectorArrayIntAddr;
	vtkPoints *pointsArray=(vtkPoints *)_pointsArrayIntAddr;
	FieldStorage::vectorFieldCellLevel_t * vectorFieldPtr=fsPtr->getVectorFieldCellLevelFieldByName(_fieldName);

	if(!vectorFieldPtr)
		return false;

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();
  set<long> globalVisitedCells;
  vector<std::tuple<long,short,short,short,float,float,float>> globalPoints;
  pUtils->setNumberOfWorkNodesAuto();
#pragma omp parallel shared(fieldDim, vectorFieldPtr, pointsArray, vectorArray, globalVisitedCells, globalPoints) // private(pt, cell, vecTmp, pt_z)
{
  set<long> visitedCells;
  vector<std::tuple<long,short,short,short,float,float,float>> localPoints;
	Point3D pt;
	CellG* cell;
	Coordinates3D<float> vecTmp;
  short pt_z;

// TODO: need OpenMP 3.0 > support on Windows to allow non-integer for loop indicies, cannot parallelize this
#pragma omp for nowait schedule(static)
	for(pt_z =0 ; pt_z<fieldDim.z ; ++pt_z)	{
    pt.z = pt_z;
		for(pt.y =0 ; pt.y<fieldDim.y ; ++pt.y) {
			for(pt.x =0 ; pt.x<fieldDim.x ; ++pt.x){
  			cell=cellFieldG->get(pt);
				if(cell){
					//check if this cell is in the set of visited Cells
					if(visitedCells.find(cell->id)!=visitedCells.end()){
						continue; //cell have been visited
					}else{
						//this is first time we visit given cell
						FieldStorage::vectorFieldCellLevelItr_t mitr=vectorFieldPtr->find(cell);
						if(mitr!=vectorFieldPtr->end()){
							vecTmp=mitr->second;
  						localPoints.push_back(make_tuple(cell->id, pt.x,pt.y,pt.z, vecTmp.x,vecTmp.y,vecTmp.z));
						}
						visitedCells.insert(cell->id);
					}
				}
			}
    }
  }
  #pragma omp critical
  {
    for (auto item : localPoints) {
      if (globalVisitedCells.find(std::get<0>(item))==globalVisitedCells.end()) {
        globalPoints.push_back(item);
        globalVisitedCells.insert(std::get<0>(item));
      }
    }
  }
#pragma omp barrier

#pragma omp sections
{
  #pragma omp section
  {
    pointsArray->SetNumberOfPoints(globalPoints.size());
  }
  #pragma omp section
  {
    vectorArray->SetNumberOfComponents(3);
    vectorArray->SetNumberOfTuples(globalPoints.size());
  }
}

#pragma omp for schedule(static)
  for (int i=0; i < globalPoints.size(); ++i) {
    auto point = globalPoints[i];
    pointsArray->SetPoint(i, std::get<1>(point), std::get<2>(point), std::get<3>(point));
    vectorArray->SetTuple3(i,std::get<4>(point),std::get<5>(point),std::get<6>(point));
  }
}
 pUtils->setNumberOfWorkNodesAuto(1);
  return true;
}


bool FieldExtractor::fillVectorFieldCellLevelData3DHex(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName) {
  vtkFloatArray *vectorArray = (vtkFloatArray *)_vectorArrayIntAddr;
  vtkPoints *pointsArray = (vtkPoints *)_pointsArrayIntAddr;

  FieldStorage::vectorFieldCellLevel_t *vectorFieldPtr = fsPtr->getVectorFieldCellLevelFieldByName(_fieldName);

  if (!vectorFieldPtr)
    return false;

  Field3D<CellG *> *cellFieldG = potts->getCellFieldG();
  Dim3D fieldDim = cellFieldG->getDim();
  set<long> globalVisitedCells;
  vector<std::tuple<long,double,double,double,float,float,float>> globalPoints;
  pUtils->setNumberOfWorkNodesAuto();
#pragma omp parallel shared(fieldDim, vectorFieldPtr, pointsArray, vectorArray, globalVisitedCells, globalPoints) // private(pt, cell, vecTmp, pt_z)
{
  set<long> visitedCells;
  vector<std::tuple<long,double,double,double,float,float,float>> localPoints;
  Point3D pt;
  vector<int> ptVec(3, 0);
  CellG *cell;
  Coordinates3D<float> vecTmp;
  short pt_z;
// TODO: need OpenMP 3.0 > support on Windows to allow non-integer for loop indicies, cannot parallelize this
#pragma omp for nowait schedule(static)
  for (pt_z = 0; pt_z < fieldDim.z; ++pt_z){
    pt.z=pt_z;
    for (pt.y = 0; pt.y < fieldDim.y; ++pt.y){
      for (pt.x = 0; pt.x < fieldDim.x; ++pt.x)
      {
        cell = cellFieldG->get(pt);
        if (cell)
        {
          // check if this cell is in the set of visited Cells
          if (visitedCells.find(cell->id) != visitedCells.end())
          {
            continue; // cell have been visited
          }
          else
          {
            // this is first time we visit given cell
            FieldStorage::vectorFieldCellLevelItr_t mitr = vectorFieldPtr->find(cell);
            if (mitr != vectorFieldPtr->end())
            {
              vecTmp = mitr->second;
              Coordinates3D<double> hexCoords = HexCoordXY(pt.x, pt.y, pt.z);
              localPoints.push_back(make_tuple(cell->id, hexCoords.x, hexCoords.y, hexCoords.z, vecTmp.x,vecTmp.y,vecTmp.z));
            }
            visitedCells.insert(cell->id);
          }
        }
      }
    }
  }
  #pragma omp critical
  {
    for (auto item : localPoints) {
      if (globalVisitedCells.find(std::get<0>(item))==globalVisitedCells.end()) {
        globalPoints.push_back(item);
        globalVisitedCells.insert(std::get<0>(item));
      }
    }
  }
#pragma omp barrier

#pragma omp sections
{
  #pragma omp section
  {
    pointsArray->SetNumberOfPoints(globalPoints.size());
  }
  #pragma omp section
  {
    vectorArray->SetNumberOfComponents(3);
    vectorArray->SetNumberOfTuples(globalPoints.size());
  }
}

#pragma omp for schedule(static)
  for (int i=0; i < globalPoints.size(); ++i) {
    auto point = globalPoints[i];
    pointsArray->SetPoint(i, std::get<1>(point), std::get<2>(point), std::get<3>(point));
    vectorArray->SetTuple3(i,std::get<4>(point),std::get<5>(point),std::get<6>(point));
  }
}
pUtils->setNumberOfWorkNodesAuto(1);
  return true;
}


vector<int> FieldExtractor::fillCellFieldData3D(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _cellIdArrayAddr, bool extractOuterShellOnly){



	vtkIntArray *cellTypeArray=(vtkIntArray *)_cellTypeArrayAddr;
	vtkLongArray *cellIdArray=(vtkLongArray *)_cellIdArrayAddr;

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim = cellFieldG->getDim();

    // if neighbor tracker is loaded we can figure out cell ids that touch medium (we call them outer cells) and render only those
    // this way we do not waste time rendering inner cells that are not seen because they are covered by outer cells.
    // this algorithm is not perfect but does significantly speed up 3D rendering

    bool neighbor_tracker_loaded = Simulator::pluginManager.isLoaded("NeighborTracker");
    //cout << "neighbor_tracker_loaded=" << neighbor_tracker_loaded << endl;
    ExtraMembersGroupAccessor<NeighborTracker> *neighborTrackerAccessorPtr;
    if (neighbor_tracker_loaded) {
        bool pluginAlreadyRegisteredFlag;
        NeighborTrackerPlugin *nTrackerPlugin = (NeighborTrackerPlugin*)Simulator::pluginManager.get("NeighborTracker", &pluginAlreadyRegisteredFlag);
        neighborTrackerAccessorPtr = nTrackerPlugin->getNeighborTrackerAccessorPtr();
    }

    std::unordered_set<long> outer_cell_ids_set;

    // to optimize drawing individual cells in 3D we may use cell shell optimization where we draw only cells that make up a cell shell opf the volume and skip inner cells that are not visible
    bool cellShellOnlyOptimization = neighbor_tracker_loaded && extractOuterShellOnly;

    if (cellShellOnlyOptimization) {

        CellInventory::cellInventoryIterator cInvItr;
        CellG * cell;
        CellInventory & cellInventory = potts->getCellInventory();
        // TODO: need OpenMP 3.0 > support on Windows to allow non-integer for loop indicies, cannot parallelize this
        for (cInvItr = cellInventory.cellInventoryBegin(); cInvItr != cellInventory.cellInventoryEnd(); ++cInvItr)
        {
            cell = cellInventory.getCell(cInvItr);
            std::set<NeighborSurfaceData > * neighborsPtr = &(neighborTrackerAccessorPtr->get(cell->extraAttribPtr)->cellNeighbors);
            set<NeighborSurfaceData>::iterator sitr;
            for (sitr = neighborsPtr->begin(); sitr != neighborsPtr->end(); ++sitr) {
                if (!sitr->neighborAddress) {
                    outer_cell_ids_set.insert(cell->id);
                    break;
                }
            }
        }
    }

    ParallelUtilsOpenMP *pUtils = sim->pUtils;
    // todo - consider separate CPU setting for graphics
    unsigned int num_work_nodes = pUtils->getMaxNumberOfWorkNodes();
    vector<unordered_set<int> > vecUsedCellTypes(num_work_nodes ? num_work_nodes : 1);

    pUtils->setNumberOfWorkNodesAuto();
#pragma omp parallel shared(vecUsedCellTypes, cellTypeArray, cellIdArray, fieldDim, outer_cell_ids_set, cellFieldG)
{
#pragma omp sections
{
  #pragma omp section
  {
    cellTypeArray->SetNumberOfValues((fieldDim.x + 2) * (fieldDim.y + 2) * (fieldDim.z + 2));
  }
  #pragma omp section
  {
    cellIdArray->SetNumberOfValues((fieldDim.x + 2) * (fieldDim.y + 2) * (fieldDim.z + 2));
  }
}

unsigned int currentWorkNodeNumber = pUtils->getCurrentWorkNodeNumber();
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
#pragma omp for schedule(static)
	for(int k =0 ; k<fieldDim.z+2 ; ++k){
    Point3D pt;
    CellG *cell;
    int type;
    long id;

    int k_offset = k * (fieldDim.y + 2) * (fieldDim.x + 2);
    for(int j =0 ; j<fieldDim.y+2 ; ++j){
      int j_offset = j * (fieldDim.x + 2);
      for (int i = 0; i < fieldDim.x + 2; ++i)
      {
        int offset = k_offset + j_offset + i;
        if(i==0 || i==fieldDim.x+1 ||j==0 || j==fieldDim.y+1 || k==0 || k==fieldDim.z+1){
					cellTypeArray->SetValue(offset, 0);
					cellIdArray->SetValue(offset, 0);
				}else{
					pt.x=i-1;
					pt.y=j-1;
					pt.z=k-1;
					cell = cellFieldG->get(pt);
					if (!cell){
						type=0;
						id=0;
					}else{
						type = cell->type;
						id = cell->id;

            vecUsedCellTypes[currentWorkNodeNumber].insert(type)      ;
//            if (usedCellTypes.find(type) == usedCellTypes.end())
//            {
//              #pragma omp critical
//              usedCellTypes.insert(type);
//            }
          }
          if (cellShellOnlyOptimization) {
              if (outer_cell_ids_set.find(id) != outer_cell_ids_set.end()) {
                  cellTypeArray->SetValue(offset, type);
                  cellIdArray->SetValue(offset, id);
              }
              else {
                  cellTypeArray->SetValue(offset, 0);
                  cellIdArray->SetValue(offset, 0);
              }
          }
          else {
              cellTypeArray->SetValue(offset, type);
              cellIdArray->SetValue(offset, id);
          }
				}
      }
    }
  }
} // omp_parallel

  unordered_set<int> usedCellTypes;
  for (auto s: vecUsedCellTypes){
      usedCellTypes.insert(s.begin(), s.end());

  }
  return vector<int>(usedCellTypes.begin(),usedCellTypes.end());
}

bool FieldExtractor::fillConFieldData3D(vtk_obj_addr_int_t _conArrayAddr ,vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _conFieldName,std::vector<int> * _typesInvisibeVec){
  vtkDoubleArray *conArray=(vtkDoubleArray *)_conArrayAddr;
	vtkIntArray *cellTypeArray=(vtkIntArray *)_cellTypeArrayAddr;

	Field3D<float> *conFieldPtr=0;
	std::map<std::string,Field3D<float>*> & fieldMap=sim->getConcentrationFieldNameMap();
	std::map<std::string,Field3D<float>*>::iterator mitr;
	mitr=fieldMap.find(_conFieldName);
	if(mitr!=fieldMap.end()){
		conFieldPtr=mitr->second;
	}

	if(!conFieldPtr)
		return false;

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();
  set<int> invisibleTypeSet;

#pragma omp parallel shared(invisibleTypeSet, conArray, cellTypeArray, conFieldPtr, cellFieldG)
  {
#pragma omp sections
    {
#pragma omp section
    {
      conArray->SetNumberOfValues((fieldDim.x + 2) * (fieldDim.y + 2) * (fieldDim.z + 2));
    }
#pragma omp section
    {
      cellTypeArray->SetNumberOfValues((fieldDim.x + 2) * (fieldDim.y + 2) * (fieldDim.z + 2));
    }
    #pragma omp section
    {
      for (std::vector<int>::iterator it = _typesInvisibeVec->begin(); it != _typesInvisibeVec->end(); ++it)
      {
        invisibleTypeSet.insert(*it);
      }
    }
  }

	Point3D pt;
	CellG *cell;
	double con;
	int type;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
#pragma omp for schedule(static)
  for (int k = 0; k < fieldDim.z + 2; ++k) {
    int k_offset = k * (fieldDim.y + 2) * (fieldDim.x + 2);
    for (int j = 0; j < fieldDim.y + 2; ++j) {
      int j_offset = j * (fieldDim.x + 2);
      for (int i = 0; i < fieldDim.x + 2; ++i) {
        int offset = k_offset + j_offset + i;
        if(i==0 || i==fieldDim.x+1 ||j==0 || j==fieldDim.y+1 || k==0 || k==fieldDim.z+1){
					conArray->SetValue(offset, 0.0);
					cellTypeArray->SetValue(offset, 0);
				}else{
					pt.x=i-1;
					pt.y=j-1;
					pt.z=k-1;
					con=conFieldPtr->get(pt);
					cell=cellFieldG->get(pt);
					if(cell)
						if(invisibleTypeSet.find(cell->type)!=invisibleTypeSet.end()){
							type=0;
						}
						else{
							type=cell->type;
						}
					else{
						type=0;
					}
					conArray->SetValue(offset, con);
          cellTypeArray->SetValue(offset, type);
        }
      }
    }
  }
}
  pUtils->setNumberOfWorkNodesAuto(1);
  return true;
}
// rwh: leave this function in until we determine we really don't want to add a boundary layer
bool FieldExtractor::fillScalarFieldData3D(vtk_obj_addr_int_t _conArrayAddr ,vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _conFieldName,std::vector<int> * _typesInvisibeVec){
  vtkDoubleArray *conArray=(vtkDoubleArray *)_conArrayAddr;
	vtkIntArray *cellTypeArray=(vtkIntArray *)_cellTypeArrayAddr;
	FieldStorage::floatField3D_t * conFieldPtr=fsPtr->getScalarFieldByName(_conFieldName);

	if(!conFieldPtr)
		return false;


	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();
  set<int> invisibleTypeSet;
  pUtils->setNumberOfWorkNodesAuto();
#pragma omp parallel shared(invisibleTypeSet, conArray, cellTypeArray, conFieldPtr, cellFieldG)
  {
#pragma omp sections
    {
#pragma omp section
      {
        conArray->SetNumberOfValues((fieldDim.x + 2) * (fieldDim.y + 2) * (fieldDim.z + 2));
      }
#pragma omp section
      {
        cellTypeArray->SetNumberOfValues((fieldDim.x + 2) * (fieldDim.y + 2) * (fieldDim.z + 2));
      }
#pragma omp section
      {
        for (std::vector<int>::iterator it = _typesInvisibeVec->begin(); it != _typesInvisibeVec->end(); ++it)
        {
          invisibleTypeSet.insert(*it);
        }
      }
    }
  Point3D pt;
  CellG *cell;
  double con;
  int type;
  // when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
#pragma omp for schedule(static)
  for (int k = 0; k < fieldDim.z + 2; ++k) {
    int k_offset = k * (fieldDim.y + 2) * (fieldDim.x + 2);
    for (int j = 0; j < fieldDim.y + 2; ++j) {
      int j_offset = j * (fieldDim.x + 2);
      for (int i = 0; i < fieldDim.x + 2; ++i) {
        int offset = k_offset + j_offset + i;
        if (i == 0 || i == fieldDim.x + 1 || j == 0 || j == fieldDim.y + 1 || k == 0 || k == fieldDim.z + 1)
        {
          conArray->SetValue(offset, 0.0);
          cellTypeArray->SetValue(offset, 0);
        }
        else
        {
          pt.x = i - 1;
          pt.y = j - 1;
          pt.z = k - 1;
          con = (*conFieldPtr)[pt.x][pt.y][pt.z];
          cell = cellFieldG->get(pt);
          if (cell)
            if (invisibleTypeSet.find(cell->type) != invisibleTypeSet.end())
            {
              type = 0;
            }
            else
            {
              type = cell->type;
            }
          else
          {
            type = 0;
          }
          conArray->SetValue(offset, con);
          cellTypeArray->SetValue(offset, type);
        }
      }
    }
  }
}
  pUtils->setNumberOfWorkNodesAuto(1);
  return true;
}

// rwh: leave this (new) function in until we determine if we want to NOT add boundary layer
// bool FieldExtractor::fillScalarFieldData3D(long _conArrayAddr ,long _cellTypeArrayAddr, std::string _conFieldName,std::vector<int> * _typesInvisibeVec){
//
//	vtkDoubleArray *conArray = (vtkDoubleArray *)_conArrayAddr;
//	vtkIntArray *cellTypeArray = (vtkIntArray *)_cellTypeArrayAddr;
//	FieldStorage::floatField3D_t * conFieldPtr = fsPtr->getScalarFieldByName(_conFieldName);
//
//	if(!conFieldPtr)
//		return false;
//
//
//	Field3D<CellG*> * cellFieldG = potts->getCellFieldG();
//	Dim3D fieldDim = cellFieldG->getDim();
//
//	conArray->SetNumberOfValues((fieldDim.x)*(fieldDim.y)*(fieldDim.z));
//	cellTypeArray->SetNumberOfValues((fieldDim.x)*(fieldDim.y)*(fieldDim.z));
//
//	set<int> invisibleTypeSet(_typesInvisibeVec->begin(),_typesInvisibeVec->end());
//
//	//for (set<int>::iterator sitr=invisibleTypeSet.begin();sitr!=invisibleTypeSet.end();++sitr){
//	//	cerr<<"invisible type="<<*sitr<<endl;
//	//}
//
//	Point3D pt;
//	CellG *cell;
//	double con;
//	int type;
//	int offset=0;
//	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
//	for(int k =0 ; k<fieldDim.z ; ++k)
//		for(int j =0 ; j<fieldDim.y ; ++j)
//			for(int i =0 ; i<fieldDim.x ; ++i){
//					pt.x=i;
//					pt.y=j;
//					pt.z=k;
//					con=(*conFieldPtr)[pt.x][pt.y][pt.z];
//					cell=cellFieldG->get(pt);
//					if(cell)
//						if(invisibleTypeSet.find(cell->type)!=invisibleTypeSet.end()){
//							type=0;
//						}
//						else{
//							type=cell->type;
//						}
//					else{
//						type=0;
//					}
//					conArray->InsertValue(offset, con);
//					cellTypeArray->InsertValue(offset, type);
//					++offset;
//			}
//			return true;
//}

bool FieldExtractor::fillScalarFieldCellLevelData3D(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _conFieldName, std::vector<int> * _typesInvisibeVec)
{
  vtkDoubleArray *conArray = (vtkDoubleArray *)_conArrayAddr;
  vtkIntArray *cellTypeArray = (vtkIntArray *)_cellTypeArrayAddr;
  FieldStorage::scalarFieldCellLevel_t *conFieldPtr = fsPtr->getScalarFieldCellLevelFieldByName(_conFieldName);

  FieldStorage::scalarFieldCellLevel_t::iterator mitr;

  if (!conFieldPtr)
    return false;

  Field3D<CellG *> *cellFieldG = potts->getCellFieldG();
  Dim3D fieldDim = cellFieldG->getDim();

  conArray->SetNumberOfValues((fieldDim.x + 2) * (fieldDim.y + 2) * (fieldDim.z + 2));
  cellTypeArray->SetNumberOfValues((fieldDim.x + 2) * (fieldDim.y + 2) * (fieldDim.z + 2));
  set<int> invisibleTypeSet;
  pUtils->setNumberOfWorkNodesAuto();
#pragma omp parallel shared(invisibleTypeSet, conArray, cellTypeArray, conFieldPtr, cellFieldG)
{
#pragma omp sections
  {
#pragma omp section
    {
      conArray->SetNumberOfValues((fieldDim.x + 2) * (fieldDim.y + 2) * (fieldDim.z + 2));
    }
#pragma omp section
    {
      cellTypeArray->SetNumberOfValues((fieldDim.x + 2) * (fieldDim.y + 2) * (fieldDim.z + 2));
    }
#pragma omp section
    {
      for (std::vector<int>::iterator it = _typesInvisibeVec->begin(); it != _typesInvisibeVec->end(); ++it)
      {
        invisibleTypeSet.insert(*it);
      }
    }
  }
  Point3D pt;
  CellG *cell;
  double con;
  int type;
  // when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
#pragma omp for schedule(static)
  for (int k = 0; k < fieldDim.z + 2; ++k) {
    int k_offset = k * (fieldDim.y + 2) * (fieldDim.x + 2);
    for (int j = 0; j < fieldDim.y + 2; ++j) {
      int j_offset = j * (fieldDim.x + 2);
      for (int i = 0; i < fieldDim.x + 2; ++i) {
        int offset = k_offset + j_offset + i;
        if (i == 0 || i == fieldDim.x + 1 || j == 0 || j == fieldDim.y + 1 || k == 0 || k == fieldDim.z + 1)
        {
          conArray->SetValue(offset, 0.0);
          cellTypeArray->SetValue(offset, 0);
        }
        else
        {
          pt.x = i - 1;
          pt.y = j - 1;
          pt.z = k - 1;

          cell = cellFieldG->get(pt);

          if (cell)
          {
            mitr = conFieldPtr->find(cell);
            if (mitr != conFieldPtr->end())
            {
              type = cell->type;
              con = mitr->second;
            }
            else
            {
              type = cell->type;
              con = 0.0;
            }
          }
          else
          {
            type = 0;
            con = 0.0;
          }
          conArray->SetValue(offset, con);
          cellTypeArray->SetValue(offset, type);
        }
      }
    }
  }
}
  pUtils->setNumberOfWorkNodesAuto(1);
  return true;
}

void FieldExtractor::setVtkObj(void *_vtkObj)
{
  cerr << "INSIDE setVtkObj" << endl;
}

void FieldExtractor::setVtkObjInt(long _vtkObjAddr)
{
  void *vPtr = (void *)_vtkObjAddr;
  cerr << "GOT THIS VOID ADDR " << vPtr << endl;
  vtkIntArray *arrayPtr = (vtkIntArray *)vPtr;
  arrayPtr->SetName("INTEGER ARRAY");
  cerr << "THIS IS NAME OF THE ARRAY=" << arrayPtr->GetName() << endl;
}

vtkIntArray *FieldExtractor::produceVtkIntArray()
{
  vtkIntArray *vtkIntArrayObj = vtkIntArray::New();
  return vtkIntArrayObj;
}

int *FieldExtractor::produceArray(int _size)
{
  return new int[_size];
}
