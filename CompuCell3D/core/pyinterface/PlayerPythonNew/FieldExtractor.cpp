    
#include "CellGraphicsData.h"
#include <iostream>
#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>
#include <CompuCell3D/plugins/ECMaterials/ECMaterialsPlugin.h>
#include <Utils/Coordinates3D.h>
#include <vtkIntArray.h>
#include <vtkLongArray.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkType.h>
#include <algorithm>
#include <cmath>

#include <vtkPythonUtil.h>
#include <vtkLookupTable.h>

using namespace std;
using namespace CompuCell3D;


#include "FieldExtractor.h"

FieldExtractor::FieldExtractor():fsPtr(0),potts(0),sim(0),ecmPlugin(0)
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

	_cellTypeArray->SetNumberOfValues((dim[1]+2)*(dim[0]+1));
	//For some reasons the points x=0 are eaten up (don't know why).
	//So we just populate empty cellIds.
	int offset=0;
	for (int i = 0 ; i< dim[0]+1 ;++i){
		_cellTypeArray->SetValue(offset, 0);
		++offset;
	}

	Point3D pt;
	vector<int> ptVec(3,0);
	CellG* cell;
	int type;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
	for(int j =0 ; j<dim[1]+1 ; ++j)
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
			_cellTypeArray->InsertValue(offset, type);
			++offset;
		}
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

    int offset = 0;

    Point3D pt;
    vector<int> ptVec(3, 0);
    CellG* cell;
    int type;
    long pc = 0;


    //when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned


    for (int j = 0; j<dim[1]; ++j)
        for (int i = 0; i<dim[0]; ++i) {
            ptVec[0] = i;
            ptVec[1] = j;
            ptVec[2] = _pos;

            pt.x = ptVec[pointOrderVec[0]];
            pt.y = ptVec[pointOrderVec[1]];
            pt.z = ptVec[pointOrderVec[2]];

            cell = cellFieldG->get(pt);
            if (!cell) {
                type = 0;
                continue;
            }
            else {
                type = cell->type;
            }


            Coordinates3D<double> coords(ptVec[0], ptVec[1], 0); // notice that we are drawing pixels from other planes on a xy plan so we use ptVec instead of pt. pt is absolute position of the point ptVec is for projection purposes

            for (int idx = 0; idx<4; ++idx) {
                Coordinates3D<double> cartesianVertex = cartesianVertices[idx] + coords;
                _pointsArray->InsertNextPoint(cartesianVertex.x, cartesianVertex.y, 0.0);
            }

            pc += 4;
            vtkIdType cellId = _cellsArray->InsertNextCell(4);
            _cellsArray->InsertCellPoint(pc - 4);
            _cellsArray->InsertCellPoint(pc - 3);
            _cellsArray->InsertCellPoint(pc - 2);
            _cellsArray->InsertCellPoint(pc - 1);

            _cellTypeArray->InsertNextValue(type);
            ++offset;
        }
    

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

	int offset=0;

	////For some reasons the points x=0 are eaten up (don't know why).
	////So we just populate empty cellIds.

	//for (int i = 0 ; i< dim[0]+1 ;++i){
	//	_cellTypeArray->SetValue(offset, 0);
	//	++offset;
	//}

	Point3D pt;
	vector<int> ptVec(3,0);
	CellG* cell;
	int type;
	long pc=0;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
	for(int j =0 ; j<dim[1] ; ++j)
		for(int i =0 ; i<dim[0] ; ++i){
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

			cell=cellFieldG->get(pt);
			if (!cell){
				type=0;
				continue;
			}else{
				type=cell->type;
			}

			Coordinates3D<double> hexCoords=HexCoordXY(pt.x,pt.y,pt.z);
			for (int idx=0 ; idx<6 ; ++idx){
			 Coordinates3D<double> hexagonVertex=hexagonVertices[idx]+hexCoords;
			 _pointsArray->InsertNextPoint(hexagonVertex.x,hexagonVertex.y,0.0);
			}
			pc+=6;
			vtkIdType cellId = _hexCellsArray->InsertNextCell(6);
			_hexCellsArray->InsertCellPoint(pc-6);
			_hexCellsArray->InsertCellPoint(pc-5);
			_hexCellsArray->InsertCellPoint(pc-4);
			_hexCellsArray->InsertCellPoint(pc-3);
			_hexCellsArray->InsertCellPoint(pc-2);
			_hexCellsArray->InsertCellPoint(pc-1);

			_cellTypeArray->InsertNextValue(type);

			++offset;
		}
}

void FieldExtractor::fillCellFieldData2DHex_old(vtk_obj_addr_int_t _cellTypeArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane ,  int _pos){
	vtkIntArray *_cellTypeArray=(vtkIntArray *)_cellTypeArrayAddr;
	vtkPoints *_pointsArray=(vtkPoints *)_pointsArrayAddr;

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

	_cellTypeArray->SetNumberOfValues((dim[1])*(dim[0]));
	_pointsArray->SetNumberOfPoints((dim[1])*(dim[0]));

	int offset=0;

	////For some reasons the points x=0 are eaten up (don't know why).
	////So we just populate empty cellIds.

	//for (int i = 0 ; i< dim[0]+1 ;++i){
	//	_cellTypeArray->SetValue(offset, 0);
	//	++offset;
	//}

	Point3D pt;
	vector<int> ptVec(3,0);
	CellG* cell;
	int type;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
	for(int j =0 ; j<dim[1] ; ++j)
		for(int i =0 ; i<dim[0] ; ++i){
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
			Coordinates3D<double> hexCoords=HexCoordXY(pt.x,pt.y,pt.z);
			_cellTypeArray->InsertValue(offset, type);
			_pointsArray->InsertPoint(offset, hexCoords.x,hexCoords.y,0.0);

			++offset;
		}
}

void FieldExtractor::fillBorderData2D(vtk_obj_addr_int_t _pointArrayAddr ,vtk_obj_addr_int_t _linesArrayAddr, std::string _plane ,  int _pos){

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

	Point3D pt;
	vector<int> ptVec(3,0);
	Point3D ptN;
	vector<int> ptNVec(3,0);

	int k=0;
	int pc=0;

	for(int i=0; i <dim[0]; ++i)
		for(int j=0; j <dim[1]; ++j){
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

			if(i > 0 && j < dim[1] ){
				ptNVec[0]=i-1;
				ptNVec[1]=j;
				ptNVec[2]=_pos;
				ptN.x=ptNVec[pointOrderVec[0]];
				ptN.y=ptNVec[pointOrderVec[1]];
				ptN.z=ptNVec[pointOrderVec[2]];
				if (cellFieldG->get(pt) != cellFieldG->get(ptN) ){
					points->InsertNextPoint(i,j,0);
					points->InsertNextPoint(i,j+1,0);
					pc+=2;
					lines->InsertNextCell(2);
					lines->InsertCellPoint(pc-2);
					lines->InsertCellPoint(pc-1);
				}
			}
			if(j > 0 && i < dim[0] ){
				ptNVec[0]=i;
				ptNVec[1]=j-1;
				ptNVec[2]=_pos;
				ptN.x=ptNVec[pointOrderVec[0]];
				ptN.y=ptNVec[pointOrderVec[1]];
				ptN.z=ptNVec[pointOrderVec[2]];
				if (cellFieldG->get(pt) != cellFieldG->get(ptN) ){
					points->InsertNextPoint(i,j,0);
					points->InsertNextPoint(i+1,j,0);
					pc+=2;
					lines->InsertNextCell(2);
					lines->InsertCellPoint(pc-2);
					lines->InsertCellPoint(pc-1);
				}
			}

			if( i < dim[0] && j < dim[1]  ){
				ptNVec[0]=i+1;
				ptNVec[1]=j;
				ptNVec[2]=_pos;
				ptN.x=ptNVec[pointOrderVec[0]];
				ptN.y=ptNVec[pointOrderVec[1]];
				ptN.z=ptNVec[pointOrderVec[2]];
				if (cellFieldG->get(pt) != cellFieldG->get(ptN) ){
					points->InsertNextPoint(i+1,j,0);
					points->InsertNextPoint(i+1,j+1,0);
					pc+=2;
					lines->InsertNextCell(2);
					lines->InsertCellPoint(pc-2);
					lines->InsertCellPoint(pc-1);
				}
			}

			if( i < dim[0] && j < dim[1]  ){
				ptNVec[0]=i;
				ptNVec[1]=j+1;
				ptNVec[2]=_pos;
				ptN.x=ptNVec[pointOrderVec[0]];
				ptN.y=ptNVec[pointOrderVec[1]];
				ptN.z=ptNVec[pointOrderVec[2]];
				if (cellFieldG->get(pt) != cellFieldG->get(ptN) ){
					points->InsertNextPoint(i,j+1,0);
					points->InsertNextPoint(i+1,j+1,0);
					pc+=2;
					lines->InsertNextCell(2);
					lines->InsertCellPoint(pc-2);
					lines->InsertCellPoint(pc-1);
				}
			}
		}
}

void FieldExtractor::fillBorderData2DHex(vtk_obj_addr_int_t _pointArrayAddr ,vtk_obj_addr_int_t _linesArrayAddr, std::string _plane ,  int _pos){
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

	Point3D pt;
	vector<int> ptVec(3,0);
	Point3D ptN;
	vector<int> ptNVec(3,0);

	int k=0;
	int pc=0;

	for(int i=0; i <dim[0]; ++i)
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
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if(pt.x-1>=0 && pt.y+1<dim[1]){
                        ptN.x=pt.x-1;
                        ptN.y=pt.y+1;
                        ptN.z=pt.z;
                        if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[5]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[0]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if( pt.y+1<dim[1]){
                        ptN.x=pt.x;
                        ptN.y=pt.y+1;
                        ptN.z=pt.z;
                        if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[0]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[1]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if( pt.x+1<dim[0]){
                        ptN.x=pt.x+1;
                        ptN.y=pt.y;
                        ptN.z=pt.z;
                        if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[1]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[2]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if( pt.y-1>=0){
                        ptN.x=pt.x;
                        ptN.y=pt.y-1;
                        ptN.z=pt.z;
                        if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[2]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[3]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if(pt.x-1>=0 && pt.y-1>=0){
                        ptN.x=pt.x-1;
                        ptN.y=pt.y-1;
                        ptN.z=pt.z;
                        if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[3]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[4]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
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
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if( pt.y+1<dim[1]){
                        ptN.x=pt.x;
                        ptN.y=pt.y+1;
                        ptN.z=pt.z;
                        if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[5]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[0]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if(pt.x+1<dim[0] && pt.y+1<dim[1]){
                        ptN.x=pt.x+1;
                        ptN.y=pt.y+1;
                        ptN.z=pt.z;
                        if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[0]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[1]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if(pt.x+1<dim[0] ){
                        ptN.x=pt.x+1;
                        ptN.y=pt.y;
                        ptN.z=pt.z;
                        if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[1]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[2]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if(pt.x+1<dim[0] && pt.y-1>=0 ){
                        ptN.x=pt.x+1;
                        ptN.y=pt.y-1;
                        ptN.z=pt.z;
                        if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[2]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[3]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if( pt.y-1>=0 ){
                        ptN.x=pt.x;
                        ptN.y=pt.y-1;
                        ptN.z=pt.z;
                        if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[3]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[4]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
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
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
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
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if( pt.y+1<dim[1]){
                        ptN.x=pt.x;
                        ptN.y=pt.y+1;
                        ptN.z=pt.z;
                        if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[5]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[0]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if( pt.x+1<dim[0]){
                        ptN.x=pt.x+1;
                        ptN.y=pt.y;
                        ptN.z=pt.z;
                        if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[1]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[2]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if( pt.y-1>=0){
                        ptN.x=pt.x;
                        ptN.y=pt.y-1;
                        ptN.z=pt.z;
                        if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[3]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[4]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if(pt.x-1>=0 && pt.y-1>=0){
                        ptN.x=pt.x+1;
                        ptN.y=pt.y+1;
                        ptN.z=pt.z;
                        if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[0]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[1]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
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
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
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
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if( pt.y+1<dim[1]){
                        ptN.x=pt.x;
                        ptN.y=pt.y+1;
                        ptN.z=pt.z;
                        if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[0]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[1]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if( pt.x+1<dim[0]){
                        ptN.x=pt.x+1;
                        ptN.y=pt.y;
                        ptN.z=pt.z;
                        if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[1]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[2]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if( pt.y-1>=0){
                        ptN.x=pt.x;
                        ptN.y=pt.y-1;
                        ptN.z=pt.z;
                        if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[2]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[3]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if(pt.x-1>=0 && pt.y-1>=0){
                        ptN.x=pt.x-1;
                        ptN.y=pt.y-1;
                        ptN.z=pt.z;
                        if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[3]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[4]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }



                }                
            
            }
            
            
		}
}

void FieldExtractor::fillClusterBorderData2D(vtk_obj_addr_int_t _pointArrayAddr ,vtk_obj_addr_int_t _linesArrayAddr, std::string _plane ,  int _pos){
        

	vtkPoints *points = (vtkPoints *)_pointArrayAddr;
	vtkCellArray * lines=(vtkCellArray *)_linesArrayAddr;

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

	Point3D pt;
	vector<int> ptVec(3,0);
	Point3D ptN;
	vector<int> ptNVec(3,0);

	int k=0;
	int pc=0;

	for(int i=0; i <dim[0]; ++i) {
//		cout << "i ="<<i<<endl;
		for(int j=0; j <dim[1]; ++j){
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

			if (cellFieldG->get(pt) == 0) continue;

			long clusterId = cellFieldG->get(pt)->clusterId;
//			cout << "j,clusterId=" << j<<","<<clusterId << endl;

			if(i > 0 && j < dim[1] ){
				ptNVec[0]=i-1;
				ptNVec[1]=j;
				ptNVec[2]=_pos;
				ptN.x=ptNVec[pointOrderVec[0]];
				ptN.y=ptNVec[pointOrderVec[1]];
				ptN.z=ptNVec[pointOrderVec[2]];
				if ((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
					points->InsertNextPoint(i,j,0);
					points->InsertNextPoint(i,j+1,0);
					pc+=2;
					lines->InsertNextCell(2);
					lines->InsertCellPoint(pc-2);
					lines->InsertCellPoint(pc-1);
				}
			}
			if(j > 0 && i < dim[0] ){
				ptNVec[0]=i;
				ptNVec[1]=j-1;
				ptNVec[2]=_pos;
				ptN.x=ptNVec[pointOrderVec[0]];
				ptN.y=ptNVec[pointOrderVec[1]];
				ptN.z=ptNVec[pointOrderVec[2]];
				if ((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
					points->InsertNextPoint(i,j,0);
					points->InsertNextPoint(i+1,j,0);
					pc+=2;
					lines->InsertNextCell(2);
					lines->InsertCellPoint(pc-2);
					lines->InsertCellPoint(pc-1);
				}
			}

			if( i < dim[0] && j < dim[1]  ){
				ptNVec[0]=i+1;
				ptNVec[1]=j;
				ptNVec[2]=_pos;
				ptN.x=ptNVec[pointOrderVec[0]];
				ptN.y=ptNVec[pointOrderVec[1]];
				ptN.z=ptNVec[pointOrderVec[2]];
				if ((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
					points->InsertNextPoint(i+1,j,0);
					points->InsertNextPoint(i+1,j+1,0);
					pc+=2;
					lines->InsertNextCell(2);
					lines->InsertCellPoint(pc-2);
					lines->InsertCellPoint(pc-1);
				}
			}

			if( i < dim[0] && j < dim[1]  ){
				ptNVec[0]=i;
				ptNVec[1]=j+1;
				ptNVec[2]=_pos;
				ptN.x=ptNVec[pointOrderVec[0]];
				ptN.y=ptNVec[pointOrderVec[1]];
				ptN.z=ptNVec[pointOrderVec[2]];
				if ((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
					points->InsertNextPoint(i,j+1,0);
					points->InsertNextPoint(i+1,j+1,0);
					pc+=2;
					lines->InsertNextCell(2);
					lines->InsertCellPoint(pc-2);
					lines->InsertCellPoint(pc-1);
				}
			}
		}
	}
}

void FieldExtractor::fillClusterBorderData2DHex(vtk_obj_addr_int_t _pointArrayAddr ,vtk_obj_addr_int_t _linesArrayAddr, std::string _plane ,  int _pos){
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

	Point3D pt;
	vector<int> ptVec(3,0);
	Point3D ptN;
	vector<int> ptNVec(3,0);

	int k=0;
	int pc=0;

	for(int i=0; i <dim[0]; ++i)
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
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if(pt.x-1>=0 && pt.y+1<dim[1]){
                        ptN.x=pt.x-1;
                        ptN.y=pt.y+1;
                        ptN.z=pt.z;
                        if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[5]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[0]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if( pt.y+1<dim[1]){
                        ptN.x=pt.x;
                        ptN.y=pt.y+1;
                        ptN.z=pt.z;
                        if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[0]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[1]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if( pt.x+1<dim[0]){
                        ptN.x=pt.x+1;
                        ptN.y=pt.y;
                        ptN.z=pt.z;
                        if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[1]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[2]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if( pt.y-1>=0){
                        ptN.x=pt.x;
                        ptN.y=pt.y-1;
                        ptN.z=pt.z;
                        if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[2]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[3]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if(pt.x-1>=0 && pt.y-1>=0){
                        ptN.x=pt.x-1;
                        ptN.y=pt.y-1;
                        ptN.z=pt.z;
                        if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[3]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[4]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
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
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if( pt.y+1<dim[1]){
                        ptN.x=pt.x;
                        ptN.y=pt.y+1;
                        ptN.z=pt.z;
                        if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[5]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[0]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if(pt.x+1<dim[0] && pt.y+1<dim[1]){
                        ptN.x=pt.x+1;
                        ptN.y=pt.y+1;
                        ptN.z=pt.z;
                        if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[0]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[1]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if(pt.x+1<dim[0] ){
                        ptN.x=pt.x+1;
                        ptN.y=pt.y;
                        ptN.z=pt.z;
                        if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[1]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[2]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if(pt.x+1<dim[0] && pt.y-1>=0 ){
                        ptN.x=pt.x+1;
                        ptN.y=pt.y-1;
                        ptN.z=pt.z;
                        if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[2]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[3]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if( pt.y-1>=0 ){
                        ptN.x=pt.x;
                        ptN.y=pt.y-1;
                        ptN.z=pt.z;
                        if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[3]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[4]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
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
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if(pt.x-1>=0 && pt.y+1<dim[1]){
                        // ptN.x=pt.x-1;
                        // ptN.y=pt.y+1;
                        ptN.x=pt.x+1;
                        ptN.y=pt.y-1;
                        
                        ptN.z=pt.z;
                        if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[2]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[3]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if( pt.y+1<dim[1]){
                        ptN.x=pt.x;
                        ptN.y=pt.y+1;
                        ptN.z=pt.z;
                        if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[5]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[0]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if( pt.x+1<dim[0]){
                        ptN.x=pt.x+1;
                        ptN.y=pt.y;
                        ptN.z=pt.z;
                        if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[1]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[2]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if( pt.y-1>=0){
                        ptN.x=pt.x;
                        ptN.y=pt.y-1;
                        ptN.z=pt.z;
                        if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[3]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[4]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if(pt.x-1>=0 && pt.y-1>=0){
                        ptN.x=pt.x+1;
                        ptN.y=pt.y+1;
                        ptN.z=pt.z;
                        if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[0]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[1]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
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
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if(pt.x-1>=0 && pt.y+1<dim[1]){
                        // ptN.x=pt.x-1;
                        // ptN.y=pt.y+1;
                        ptN.x=pt.x-1;
                        ptN.y=pt.y+1;
                        
                        ptN.z=pt.z;
                        if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[5]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[0]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if( pt.y+1<dim[1]){
                        ptN.x=pt.x;
                        ptN.y=pt.y+1;
                        ptN.z=pt.z;
                        if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[0]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[1]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if( pt.x+1<dim[0]){
                        ptN.x=pt.x+1;
                        ptN.y=pt.y;
                        ptN.z=pt.z;
                        if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[1]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[2]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if( pt.y-1>=0){
                        ptN.x=pt.x;
                        ptN.y=pt.y-1;
                        ptN.z=pt.z;
                        if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[2]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[3]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }
                    if(pt.x-1>=0 && pt.y-1>=0){
                        ptN.x=pt.x-1;
                        ptN.y=pt.y-1;
                        ptN.z=pt.z;
                        if((cellFieldG->get(ptN) == 0) || clusterId != cellFieldG->get(ptN)->clusterId ){
                            Coordinates3D<double> hexCoordsP1=hexagonVertices[3]+hexCoords;
                            Coordinates3D<double> hexCoordsP2=hexagonVertices[4]+hexCoords;
                            points->InsertNextPoint(hexCoordsP1.x,hexCoordsP1.y,0.0);
                            points->InsertNextPoint(hexCoordsP2.x,hexCoordsP2.y,0.0);
                            pc+=2;
                            lines->InsertNextCell(2);
                            lines->InsertCellPoint(pc-2);
                            lines->InsertCellPoint(pc-1);
                        }
                    }

                
                
                }
            
            
            }
            
            
		}
}

void FieldExtractor::fillCentroidData2D(vtk_obj_addr_int_t _pointArrayAddr ,vtk_obj_addr_int_t _linesArrayAddr, std::string _plane ,  int _pos){
//	cerr << "FieldExtractor::fillCentroidData2D============    numCells="<< potts->getNumCells() <<endl;
	CellInventory *cellInventoryPtr = &potts->getCellInventory();
	CellInventory::cellInventoryIterator cInvItr;
	CellG * cell;

	float x,y,z;

	vtkPoints *points = (vtkPoints *)_pointArrayAddr;
	vtkCellArray * lines=(vtkCellArray *)_linesArrayAddr;

	int ptCount=0;
	for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr ){
		cell = cellInventoryPtr->getCell(cInvItr);
//		cerr << "numerator CM(x,y,z) ="<<cell->xCM<<","<<cell->yCM<<","<<cell->zCM <<"; volume="<<(float)cell->volume<<endl;
		float cellVol = (float)cell->volume;
		if (!cell->volume) {
//		  cerr <<"      centroid= "<<cell->xCM/cellVol<<","<<cell->yCM/cellVol<<","<<cell->zCM/cellVol <<endl;
//		  cerr << "FieldExtractor::fillBorderData2D:  cell volume is 0 -- exit";
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

bool FieldExtractor::fillConFieldData2DHex(vtk_obj_addr_int_t _conArrayAddr,vtk_obj_addr_int_t _hexCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos){
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


	int offset=0;

	Point3D pt;
	vector<int> ptVec(3,0);

	double con;
	long pc=0;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
	for(int j =0 ; j<dim[1] ; ++j)
		for(int i =0 ; i<dim[0] ; ++i){
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
			for (int idx=0 ; idx<6 ; ++idx){
			 Coordinates3D<double> hexagonVertex=hexagonVertices[idx]+hexCoords;
			 _pointsArray->InsertNextPoint(hexagonVertex.x,hexagonVertex.y,0.0);
			}
			pc+=6;
			vtkIdType cellId = _hexCellsArray->InsertNextCell(6);
			_hexCellsArray->InsertCellPoint(pc-6);
			_hexCellsArray->InsertCellPoint(pc-5);
			_hexCellsArray->InsertCellPoint(pc-4);
			_hexCellsArray->InsertCellPoint(pc-3);
			_hexCellsArray->InsertCellPoint(pc-2);
			_hexCellsArray->InsertCellPoint(pc-1);

			conArray->InsertNextValue( con);
		}
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

    
	int offset=0;

	Point3D pt;
	vector<int> ptVec(3,0);

	double con;
	long pc=0;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
     
     
	for(int j =0 ; j<dim[1] ; ++j)
		for(int i =0 ; i<dim[0] ; ++i){
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

			if (i==dim[0] || j==dim[1]){
				con=0.0;
			}else{
				con = con = conFieldPtr->get(pt);
			}
            
            Coordinates3D<double> coords(ptVec[0],ptVec[1],0); // notice that we are drawing pixels from other planes on a xy plan so we use ptVec instead of pt. pt is absolute position of the point ptVec is for projection purposes
            
			for (int idx=0 ; idx<4 ; ++idx){
			  Coordinates3D<double> cartesianVertex=cartesianVertices[idx]+coords; 
 			 _pointsArray->InsertNextPoint(cartesianVertex.x,cartesianVertex.y,0.0);
			}               
            
			pc+=4;
			vtkIdType cellId = _cartesianCellsArray->InsertNextCell(4);
			_cartesianCellsArray->InsertCellPoint(pc-4);
			_cartesianCellsArray->InsertCellPoint(pc-3);
			_cartesianCellsArray->InsertCellPoint(pc-2);
			_cartesianCellsArray->InsertCellPoint(pc-1);

			conArray->InsertNextValue( con);
			++offset;
		}
        
		return true;
}


bool FieldExtractor::fillScalarFieldData2DHex(vtk_obj_addr_int_t _conArrayAddr,vtk_obj_addr_int_t _hexCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){
	vtkCellArray * _hexCellsArray=(vtkCellArray*)_hexCellsArrayAddr;
	vtkPoints *_pointsArray=(vtkPoints *)_pointsArrayAddr;

	FieldStorage::floatField3D_t * conFieldPtr=fsPtr->getScalarFieldByName(_conFieldName); 

	if (!conFieldPtr) {
		if (!ecmPlugin) {
			return false;
		}
		else { 
			int ECMaterialIndex;
			ECMaterialIndex = ecmPlugin->getECMaterialIndexByName(_conFieldName);
			if (ECMaterialIndex < 0) { return false; }
			fillECMaterialFieldData2DHex(_conArrayAddr, _hexCellsArrayAddr, _pointsArrayAddr, _plane, _pos, ECMaterialIndex);
			return true;
		}
	}

	vtkDoubleArray *conArray = (vtkDoubleArray *)_conArrayAddr;

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


	int offset=0;

	Point3D pt;
	vector<int> ptVec(3,0);

	double con;
	long pc=0;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
	for(int j =0 ; j<dim[1] ; ++j)
		for(int i =0 ; i<dim[0] ; ++i){
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

			for (int idx=0 ; idx<6 ; ++idx){
			 Coordinates3D<double> hexagonVertex=hexagonVertices[idx]+hexCoords;
 			 _pointsArray->InsertNextPoint(hexagonVertex.x,hexagonVertex.y,0.0);
			}
			pc+=6;
			vtkIdType cellId = _hexCellsArray->InsertNextCell(6);
			_hexCellsArray->InsertCellPoint(pc-6);
			_hexCellsArray->InsertCellPoint(pc-5);
			_hexCellsArray->InsertCellPoint(pc-4);
			_hexCellsArray->InsertCellPoint(pc-3);
			_hexCellsArray->InsertCellPoint(pc-2);
			_hexCellsArray->InsertCellPoint(pc-1);

			conArray->InsertNextValue( con);
			++offset;
		}
        
		return true;
}

bool FieldExtractor::fillScalarFieldData2DCartesian(vtk_obj_addr_int_t _conArrayAddr,vtk_obj_addr_int_t _cartesianCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){
	vtkCellArray * _cartesianCellsArray=(vtkCellArray*)_cartesianCellsArrayAddr;
	vtkPoints *_pointsArray=(vtkPoints *)_pointsArrayAddr;

	FieldStorage::floatField3D_t * conFieldPtr=fsPtr->getScalarFieldByName(_conFieldName); 

	if (!conFieldPtr) {
		if (!ecmPlugin) {
			return false;
		}
		else {
			int ECMaterialIndex;
			ECMaterialIndex = ecmPlugin->getECMaterialIndexByName(_conFieldName);
			if (ECMaterialIndex < 0) { return false; }
			fillECMaterialData2DCartesian(_conArrayAddr, _cartesianCellsArrayAddr, _pointsArrayAddr, _plane, _pos, ECMaterialIndex);
			return true;
		}
	}

	vtkDoubleArray *conArray = (vtkDoubleArray *)_conArrayAddr;

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


          
    
	int offset=0;

	Point3D pt;
	vector<int> ptVec(3,0);

	double con;
	long pc=0;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
    
    
	for(int j =0 ; j<dim[1] ; ++j)
		for(int i =0 ; i<dim[0] ; ++i){
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
			for (int idx=0 ; idx<4 ; ++idx){
			  Coordinates3D<double> cartesianVertex=cartesianVertices[idx]+coords; 
 			 _pointsArray->InsertNextPoint(cartesianVertex.x,cartesianVertex.y,0.0);
			}               
            
			pc+=4;
			vtkIdType cellId = _cartesianCellsArray->InsertNextCell(4);
			_cartesianCellsArray->InsertCellPoint(pc-4);
			_cartesianCellsArray->InsertCellPoint(pc-3);
			_cartesianCellsArray->InsertCellPoint(pc-2);
			_cartesianCellsArray->InsertCellPoint(pc-1);

			conArray->InsertNextValue( con);
			++offset;
		}
        
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

	int offset=0;

	Point3D pt;
	vector<int> ptVec(3,0);

	CellG *cell;
	double con;
	long pc=0;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
	for(int j =0 ; j<dim[1] ; ++j)
		for(int i =0 ; i<dim[0] ; ++i){
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
			for (int idx=0 ; idx<6 ; ++idx){
			 Coordinates3D<double> hexagonVertex=hexagonVertices[idx]+hexCoords;
 			 _pointsArray->InsertNextPoint(hexagonVertex.x,hexagonVertex.y,0.0);
			}
			pc+=6;
			vtkIdType cellId = _hexCellsArray->InsertNextCell(6);
			_hexCellsArray->InsertCellPoint(pc-6);
			_hexCellsArray->InsertCellPoint(pc-5);
			_hexCellsArray->InsertCellPoint(pc-4);
			_hexCellsArray->InsertCellPoint(pc-3);
			_hexCellsArray->InsertCellPoint(pc-2);
			_hexCellsArray->InsertCellPoint(pc-1);

			conArray->InsertNextValue( con);

			++offset;
		}
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


    
	int offset=0;

	Point3D pt;
	vector<int> ptVec(3,0);

	CellG *cell;
	double con;
	long pc=0;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
    
	for(int j =0 ; j<dim[1] ; ++j)
		for(int i =0 ; i<dim[0] ; ++i){
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
			for (int idx=0 ; idx<4 ; ++idx){
			  Coordinates3D<double> cartesianVertex=cartesianVertices[idx]+coords; 
 			 _pointsArray->InsertNextPoint(cartesianVertex.x,cartesianVertex.y,0.0);
			}            
            
			pc+=4;
			vtkIdType cellId = _cartesianCellsArray->InsertNextCell(4);
			_cartesianCellsArray->InsertCellPoint(pc-4);
			_cartesianCellsArray->InsertCellPoint(pc-3);
			_cartesianCellsArray->InsertCellPoint(pc-2);
			_cartesianCellsArray->InsertCellPoint(pc-1);

			conArray->InsertNextValue( con);
			++offset;
		}
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
	int offset=0;
	for (int i = 0 ; i< dim[0]+1 ;++i){
		conArray->SetValue(offset, 0.0);
		++offset;
	}

	Point3D pt;
	vector<int> ptVec(3,0);

	double con;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
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

			conArray->SetValue(offset, con);
			++offset;
		}
		return true;
}

bool FieldExtractor::fillScalarFieldData2D(vtk_obj_addr_int_t _conArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos){
	FieldStorage::floatField3D_t * conFieldPtr=fsPtr->getScalarFieldByName(_conFieldName); 

	if (!conFieldPtr){
		if (!ecmPlugin) {
			return false;
		}
		else {
			int ECMaterialIndex;
			ECMaterialIndex = ecmPlugin->getECMaterialIndexByName(_conFieldName);
			if (ECMaterialIndex < 0) { return false; }
			fillECMaterialFieldData2D(_conArrayAddr, _plane, _pos, ECMaterialIndex);
			return true;
		}
	}

	vtkDoubleArray *conArray = (vtkDoubleArray *)_conArrayAddr;

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
	int offset=0;
	for (int i = 0 ; i< dim[0]+1 ;++i){
		conArray->SetValue(offset, 0.0);
		++offset;
	}

	Point3D pt;
	vector<int> ptVec(3,0);

	double con;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
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
				con = (*conFieldPtr)[pt.x][pt.y][pt.z];
			}
			conArray->SetValue(offset, con);
			++offset;
		}
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
	int offset=0;
	for (int i = 0 ; i< dim[0]+1 ;++i){
		conArray->SetValue(offset, 0.0);
		++offset;
	}

	Point3D pt;
	vector<int> ptVec(3,0);
	CellG* cell;

	double con;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
	for(int j =0 ; j<dim[1]+1 ; ++j)
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
			conArray->SetValue(offset, con);
			++offset;
		}
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

	Point3D pt;
	vector<int> ptVec(3,0);
	CellG* cell;
	Coordinates3D<float> vecTmp;
	float  vecTmpCoord[3] ;
        float x,y,z;
	double con;

	int offset=0;

	for(int j =0 ; j<dim[1] ; ++j)
		for(int i =0 ; i<dim[0] ; ++i){
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
                        
// 			vecTmp=(*vectorFieldPtr)[pt.x][pt.y][pt.z];
// 			vecTmpCoord[0]=vecTmp.x;
// 			vecTmpCoord[1]=vecTmp.y;
// 			vecTmpCoord[2]=vecTmp.z;

			if(x!=0.0 || y!=0.0 || z!=0.0){
				pointsArray->InsertPoint(offset,ptVec[0],ptVec[1],0);
				vectorArray->InsertTuple3(offset,vecTmpCoord[pointOrderVec[0]],vecTmpCoord[pointOrderVec[1]],0);
				++offset;
			}
		}
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

	Point3D pt;
	vector<int> ptVec(3,0);
	CellG* cell;
	Coordinates3D<float> vecTmp;
	float  vecTmpCoord[3] ;
	double con;
        float x,y,z;
	int offset=0;
	for(int j =0 ; j<dim[1] ; ++j)
		for(int i =0 ; i<dim[0] ; ++i){
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];
                        
                        x=(*vectorFieldPtr)[pt.x][pt.y][pt.z][0];
                        y=(*vectorFieldPtr)[pt.x][pt.y][pt.z][1];
                        z=(*vectorFieldPtr)[pt.x][pt.y][pt.z][2];
                        
// 			vecTmp=(*vectorFieldPtr)[pt.x][pt.y][pt.z];
// 			vecTmpCoord[0]=vecTmp.x;
// 			vecTmpCoord[1]=vecTmp.y;
// 			vecTmpCoord[2]=vecTmp.z;
                    
                        vecTmpCoord[0]=x;
                        vecTmpCoord[1]=y;
                        vecTmpCoord[2]=z;
                    
			if(x!=0.0 || y!=0.0 || z!=0.0){
				Coordinates3D<double> hexCoords=HexCoordXY(pt.x,pt.y,pt.z);
				pointsArray->InsertPoint(offset, hexCoords.x,hexCoords.y,0.0);
				
				vectorArray->InsertTuple3(offset,vecTmpCoord[pointOrderVec[0]],vecTmpCoord[pointOrderVec[1]],0);
				++offset;
			}
		}
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

	Point3D pt;
	vector<int> ptVec(3,0);
	CellG* cell;
	Coordinates3D<float> vecTmp;

        float x,y,z;
        
	int offset=0;
	for(pt.z =0 ; pt.z<fieldDim.z ; ++pt.z)	
		for(pt.y =0 ; pt.y<fieldDim.y ; ++pt.y)
			for(pt.x =0 ; pt.x<fieldDim.x ; ++pt.x){
// 				vecTmp=(*vectorFieldPtr)[pt.x][pt.y][pt.z];
                                x=(*vectorFieldPtr)[pt.x][pt.y][pt.z][0];
                                y=(*vectorFieldPtr)[pt.x][pt.y][pt.z][1];
                                z=(*vectorFieldPtr)[pt.x][pt.y][pt.z][2];                                
				if(x!=0.0 || y!=0.0 || z!=0.0){
					pointsArray->InsertPoint(offset,pt.x,pt.y,pt.z);
					vectorArray->InsertTuple3(offset,x,y,z);
					++offset;
				}
			}
			return true;
}

bool FieldExtractor::fillVectorFieldCellLevelData2D(vtk_obj_addr_int_t _pointsArrayIntAddr,vtk_obj_addr_int_t _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos){
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

	Point3D pt;
	vector<int> ptVec(3,0);
	CellG* cell;
	Coordinates3D<float> vecTmp;
	float  vecTmpCoord[3] ;

	int offset=0;

	for(int j =0 ; j<dim[1] ; ++j)
		for(int i =0 ; i<dim[0] ; ++i){
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

			cell=cellFieldG->get(pt);

			if(cell){
				//check if this cell is in the set of visited Cells
				if(visitedCells.find(cell)!=visitedCells.end()){
					continue; //cell have been visited 
				}else{
					//this is first time we visit given cell
					FieldStorage::vectorFieldCellLevelItr_t mitr=vectorFieldPtr->find(cell);
					if(mitr!=vectorFieldPtr->end()){
						vecTmp=mitr->second;
						vecTmpCoord[0]=vecTmp.x;
						vecTmpCoord[1]=vecTmp.y;
						vecTmpCoord[2]=vecTmp.z;
						
						pointsArray->InsertPoint(offset,ptVec[0],ptVec[1],0);						
						vectorArray->InsertTuple3(offset,vecTmpCoord[pointOrderVec[0]],vecTmpCoord[pointOrderVec[1]],0);
						++offset;
					}
					visitedCells.insert(cell);
				}
			}

		}
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

	Point3D pt;
	vector<int> ptVec(3,0);
	CellG* cell;
	Coordinates3D<float> vecTmp;
	float  vecTmpCoord[3] ;

	int offset=0;

	for(int j =0 ; j<dim[1] ; ++j)
		for(int i =0 ; i<dim[0] ; ++i){
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

			cell=cellFieldG->get(pt);

			if(cell){
				//check if this cell is in the set of visited Cells
				if(visitedCells.find(cell)!=visitedCells.end()){
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
						pointsArray->InsertPoint(offset, hexCoords.x,hexCoords.y,0.0);

						vectorArray->InsertTuple3(offset,vecTmpCoord[pointOrderVec[0]],vecTmpCoord[pointOrderVec[1]],0);
						++offset;
					}
					visitedCells.insert(cell);
				}
			}
		}
		return true;
}

bool FieldExtractor::fillVectorFieldCellLevelData3D(vtk_obj_addr_int_t _pointsArrayIntAddr,vtk_obj_addr_int_t _vectorArrayIntAddr,std::string _fieldName){
	vtkFloatArray * vectorArray=(vtkFloatArray *)_vectorArrayIntAddr;
	vtkPoints *pointsArray=(vtkPoints *)_pointsArrayIntAddr;

	set<CellG*> visitedCells;

	FieldStorage::vectorFieldCellLevel_t * vectorFieldPtr=fsPtr->getVectorFieldCellLevelFieldByName(_fieldName); 

	if(!vectorFieldPtr)
		return false;

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

	Point3D pt;
	vector<int> ptVec(3,0);
	CellG* cell;
	Coordinates3D<float> vecTmp;

	int offset=0;
	for(pt.z =0 ; pt.z<fieldDim.z ; ++pt.z)	
		for(pt.y =0 ; pt.y<fieldDim.y ; ++pt.y)
			for(pt.x =0 ; pt.x<fieldDim.x ; ++pt.x){

				cell=cellFieldG->get(pt);

				if(cell){
					//check if this cell is in the set of visited Cells
					if(visitedCells.find(cell)!=visitedCells.end()){
						continue; //cell have been visited 
					}else{
						//this is first time we visit given cell
						FieldStorage::vectorFieldCellLevelItr_t mitr=vectorFieldPtr->find(cell);
						if(mitr!=vectorFieldPtr->end()){
							vecTmp=mitr->second;
							pointsArray->InsertPoint(offset,pt.x,pt.y,pt.z);
							vectorArray->InsertTuple3(offset,vecTmp.x,vecTmp.y,vecTmp.z);
							++offset;
						}
						visitedCells.insert(cell);
					}
				}			
			}
			return true;
}


vector<int> FieldExtractor::fillCellFieldData3D(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _cellIdArrayAddr){
	set<int> usedCellTypes;
	vtkIntArray *cellTypeArray=(vtkIntArray *)_cellTypeArrayAddr;
	vtkLongArray *cellIdArray=(vtkLongArray *)_cellIdArrayAddr;

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim = cellFieldG->getDim();

    // if neighbor tracker is loaded we can figure out cell ids that touch medium (we call them outer cells) and render only those
    // this way we do not waste time rendering inner cells that are not seen because they are covered by outer cells. 
    // this algorithm is not perfect but does significantly speed up 3D rendering

    bool neighbor_tracker_loaded = Simulator::pluginManager.isLoaded("NeighborTracker");
    //cout << "neighbor_tracker_loaded=" << neighbor_tracker_loaded << endl;
    BasicClassAccessor<NeighborTracker> *neighborTrackerAccessorPtr;
    if (neighbor_tracker_loaded) {
        bool pluginAlreadyRegisteredFlag;
        NeighborTrackerPlugin *nTrackerPlugin = (NeighborTrackerPlugin*)Simulator::pluginManager.get("NeighborTracker", &pluginAlreadyRegisteredFlag);
        neighborTrackerAccessorPtr = nTrackerPlugin->getNeighborTrackerAccessorPtr();
    }

    std::unordered_set<long> outer_cell_ids_set;
    if (neighbor_tracker_loaded) {
        
        CellInventory::cellInventoryIterator cInvItr;
        CellG * cell;
        std::set<NeighborSurfaceData > * neighborData;
        CellInventory & cellInventory = potts->getCellInventory();

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
    
	cellTypeArray->SetNumberOfValues((fieldDim.x+2)*(fieldDim.y+2)*(fieldDim.z+2));	
	cellIdArray->SetNumberOfValues((fieldDim.x+2)*(fieldDim.y+2)*(fieldDim.z+2));

	Point3D pt;
	CellG* cell;
	int type;
	long id;
	int offset=0;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
	for(int k =0 ; k<fieldDim.z+2 ; ++k)
		for(int j =0 ; j<fieldDim.y+2 ; ++j)
			for(int i =0 ; i<fieldDim.x+2 ; ++i){
				if(i==0 || i==fieldDim.x+1 ||j==0 || j==fieldDim.y+1 || k==0 || k==fieldDim.z+1){
					cellTypeArray->InsertValue(offset, 0);
					cellIdArray->InsertValue(offset, 0);
					++offset;
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
						usedCellTypes.insert(type);
					}
                    if (neighbor_tracker_loaded) {
                        if (outer_cell_ids_set.find(id) != outer_cell_ids_set.end()) {
                            cellTypeArray->InsertValue(offset, type);
                            cellIdArray->InsertValue(offset, id);
                            ++offset;
                        }
                        else {
                            cellTypeArray->InsertValue(offset, 0);
                            cellIdArray->InsertValue(offset, 0);
                            ++offset;

                        }

                    }
                    else {
                        cellTypeArray->InsertValue(offset, type);
                        cellIdArray->InsertValue(offset, id);
                        ++offset;
                    }
				}
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

	conArray->SetNumberOfValues((fieldDim.x+2)*(fieldDim.y+2)*(fieldDim.z+2));
	cellTypeArray->SetNumberOfValues((fieldDim.x+2)*(fieldDim.y+2)*(fieldDim.z+2));

	set<int> invisibleTypeSet(_typesInvisibeVec->begin(),_typesInvisibeVec->end());

	for (set<int>::iterator sitr=invisibleTypeSet.begin();sitr!=invisibleTypeSet.end();++sitr){
		cerr<<"invisible type="<<*sitr<<endl;
	}

	Point3D pt;
	CellG *cell;
	double con;
	int type;
	int offset=0;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
	for(int k =0 ; k<fieldDim.z+2 ; ++k)
		for(int j =0 ; j<fieldDim.y+2 ; ++j)
			for(int i =0 ; i<fieldDim.x+2 ; ++i){
				if(i==0 || i==fieldDim.x+1 ||j==0 || j==fieldDim.y+1 || k==0 || k==fieldDim.z+1){
					conArray->InsertValue(offset, 0.0);
					cellTypeArray->InsertValue(offset, 0);
					++offset;
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
					conArray->InsertValue(offset, con);
					cellTypeArray->InsertValue(offset, type);
					++offset;
				}
			}
			return true;
}
// rwh: leave this function in until we determine we really don't want to add a boundary layer
bool FieldExtractor::fillScalarFieldData3D(vtk_obj_addr_int_t _conArrayAddr ,vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _conFieldName,std::vector<int> * _typesInvisibeVec){

	vtkIntArray *cellTypeArray=(vtkIntArray *)_cellTypeArrayAddr;
	FieldStorage::floatField3D_t * conFieldPtr=fsPtr->getScalarFieldByName(_conFieldName);

	if (!conFieldPtr) {
		if (!ecmPlugin) {
			return false;
		}
		else {
			int ECMaterialIndex;
			ECMaterialIndex = ecmPlugin->getECMaterialIndexByName(_conFieldName);
			if (ECMaterialIndex < 0) { return false; }
			fillECMaterialFieldData3D(_conArrayAddr, ECMaterialIndex);
			return true;
		}
	}

	vtkDoubleArray *conArray = (vtkDoubleArray *)_conArrayAddr;

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

	conArray->SetNumberOfValues((fieldDim.x+2)*(fieldDim.y+2)*(fieldDim.z+2));
	cellTypeArray->SetNumberOfValues((fieldDim.x+2)*(fieldDim.y+2)*(fieldDim.z+2));

	set<int> invisibleTypeSet(_typesInvisibeVec->begin(),_typesInvisibeVec->end());

	//for (set<int>::iterator sitr=invisibleTypeSet.begin();sitr!=invisibleTypeSet.end();++sitr){
	//	cerr<<"invisible type="<<*sitr<<endl;
	//}

	Point3D pt;
	CellG *cell;
	double con;
	int type;
	int offset=0;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
	for(int k =0 ; k<fieldDim.z+2 ; ++k)
		for(int j =0 ; j<fieldDim.y+2 ; ++j)
			for(int i =0 ; i<fieldDim.x+2 ; ++i){
				if(i==0 || i==fieldDim.x+1 ||j==0 || j==fieldDim.y+1 || k==0 || k==fieldDim.z+1){
					conArray->InsertValue(offset, 0.0);
					cellTypeArray->InsertValue(offset, 0);
					++offset;
				}else{
					pt.x=i-1;
					pt.y=j-1;
					pt.z=k-1;
					con=(*conFieldPtr)[pt.x][pt.y][pt.z];
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
					conArray->InsertValue(offset, con);
					cellTypeArray->InsertValue(offset, type);
					++offset;
				}
			}
			return true;
}

// rwh: leave this (new) function in until we determine if we want to NOT add boundary layer
//bool FieldExtractor::fillScalarFieldData3D(long _conArrayAddr ,long _cellTypeArrayAddr, std::string _conFieldName,std::vector<int> * _typesInvisibeVec){
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

bool FieldExtractor::fillScalarFieldCellLevelData3D(vtk_obj_addr_int_t _conArrayAddr ,vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _conFieldName,std::vector<int> * _typesInvisibeVec){

	vtkDoubleArray *conArray=(vtkDoubleArray *)_conArrayAddr;
	vtkIntArray *cellTypeArray=(vtkIntArray *)_cellTypeArrayAddr;
	FieldStorage::scalarFieldCellLevel_t * conFieldPtr=fsPtr->getScalarFieldCellLevelFieldByName(_conFieldName); 

	FieldStorage::scalarFieldCellLevel_t::iterator mitr;

	if(!conFieldPtr)
		return false;

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

	conArray->SetNumberOfValues((fieldDim.x+2)*(fieldDim.y+2)*(fieldDim.z+2));
	cellTypeArray->SetNumberOfValues((fieldDim.x+2)*(fieldDim.y+2)*(fieldDim.z+2));

	set<int> invisibleTypeSet(_typesInvisibeVec->begin(),_typesInvisibeVec->end());

	//for (set<int>::iterator sitr=invisibleTypeSet.begin();sitr!=invisibleTypeSet.end();++sitr){
	//	cerr<<"invisible type="<<*sitr<<endl;
	//}

	Point3D pt;
	CellG *cell;
	double con;
	int type;
	int offset=0;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
	for(int k =0 ; k<fieldDim.z+2 ; ++k)
		for(int j =0 ; j<fieldDim.y+2 ; ++j)
			for(int i =0 ; i<fieldDim.x+2 ; ++i){
				if(i==0 || i==fieldDim.x+1 ||j==0 || j==fieldDim.y+1 || k==0 || k==fieldDim.z+1){
					conArray->InsertValue(offset, 0.0);
					cellTypeArray->InsertValue(offset, 0);
					++offset;
				}else{
					pt.x=i-1;
					pt.y=j-1;
					pt.z=k-1;

					cell=cellFieldG->get(pt);

					if(cell){
						mitr=conFieldPtr->find(cell);
						if(mitr!=conFieldPtr->end()){
							type=cell->type;
							con=mitr->second;
						}else{
							type=cell->type;
							con=0.0;
						}
					}else{
						type=0;		
						con=0.0;
					}
					conArray->InsertValue(offset, con);
					cellTypeArray->InsertValue(offset, type);
					++offset;
				}
			}
			return true;
}

void FieldExtractor::initECMaterials(ECMaterialsPlugin *_ecmPlugin) {
	ecmPlugin = _ecmPlugin;
}

void FieldExtractor::extractECMaterialField() {
	Field3D<CellG*> * cellFieldG = potts->getCellFieldG();
	Dim3D fieldDim = cellFieldG->getDim();
	Point3D pt;

	unsigned int numberOfMaterials = ecmPlugin->getNumberOfMaterials();
	Field3D<ECMaterialsData *> *ECMaterialsField = ecmPlugin->getECMaterialField();
	CellG *cell;

	for (pt.x = 0; pt.x < fieldDim.x; ++pt.x)
		for (pt.y = 0; pt.y < fieldDim.y; ++pt.y)
			for (pt.z = 0; pt.z < fieldDim.z; ++pt.z) {
				cell = cellFieldG->get(pt);
				if (!cell) { fsPtr->field3DECGraphicsData[pt.x][pt.y][pt.z] = ECMaterialsField->get(pt)->ECMaterialsQuantityVec; }
			}
}

void FieldExtractor::fillECMaterialFieldData2D(vtk_obj_addr_int_t _ecmQuantityArrayAddr, std::string _plane, int _pos, int _compSel) {

	Field3D<CellG*> * cellFieldG = potts->getCellFieldG();
	Dim3D fieldDim = cellFieldG->getDim();

	Field3D<ECMaterialsData *> *ECMaterialsField = ecmPlugin->getECMaterialField();

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

	int numberOfMaterials;
	std::vector<int> compSelVec;
	if (_compSel < 0) {
		numberOfMaterials = (int)ecmPlugin->getNumberOfMaterials();
		for (int i = 0; i < numberOfMaterials; ++i) { compSelVec.push_back(i); }
	}
	else {
		numberOfMaterials = 1;
		compSelVec.push_back(_compSel);
	}

	vtkDoubleArray *_quantityArray = (vtkDoubleArray *)_ecmQuantityArrayAddr;
	int numberOfTuples = dim[1] * dim[0];
	_quantityArray->SetNumberOfComponents(numberOfMaterials);
	_quantityArray->SetNumberOfTuples(numberOfTuples);
	
	int offset = 0;
	
	Point3D pt;
	vector<int> ptVec(3, 0);
	CellG* cell;
	std::vector<float> ECMaterialsQuantityVec;
	float thisQty;
	double thisQtyDbl;

	for (int j = 0; j<dim[1]; ++j)
		for (int i = 0; i<dim[0]; ++i) {
			ptVec[0] = i;
			ptVec[1] = j;
			ptVec[2] = _pos;

			pt.x = ptVec[pointOrderVec[0]];
			pt.y = ptVec[pointOrderVec[1]];
			pt.z = ptVec[pointOrderVec[2]];

			cell = cellFieldG->get(pt);

			if (!cell) {
				ECMaterialsQuantityVec = ECMaterialsField->get(pt)->ECMaterialsQuantityVec;
				for (int k = 0; k < numberOfMaterials; ++k) {
					_quantityArray->SetComponent(offset, k, (double)ECMaterialsQuantityVec[compSelVec[k]]);
				}
			}
			else {
				for (int k = 0; k < numberOfMaterials; ++k) {_quantityArray->SetComponent(offset, k, 0.0);}
			}
			++offset;
		}
}

void FieldExtractor::fillECMaterialFieldData2DHex(vtk_obj_addr_int_t _ecmQuantityArrayAddr, vtk_obj_addr_int_t _hexCellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane, int _pos, int _compSel) {
	vtkPoints *_pointsArray = (vtkPoints *)_pointsArrayAddr;
	vtkCellArray * _hexCellsArray = (vtkCellArray*)_hexCellsArrayAddr;

	Field3D<CellG*> * cellFieldG = potts->getCellFieldG();
	Dim3D fieldDim = cellFieldG->getDim();

	Field3D<ECMaterialsData *> *ECMaterialsField = ecmPlugin->getECMaterialField();

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

	int numberOfMaterials;
	std::vector<int> compSelVec;
	if (_compSel < 0) {
		numberOfMaterials = (int)ecmPlugin->getNumberOfMaterials();
		for (int i = 0; i < numberOfMaterials; ++i) { compSelVec.push_back(i); }
	}
	else {
		numberOfMaterials = 1;
		compSelVec.push_back(_compSel);
	}

	vtkDoubleArray *_quantityArray = (vtkDoubleArray *)_ecmQuantityArrayAddr;
	_quantityArray->SetNumberOfComponents(numberOfMaterials);
	_quantityArray->SetNumberOfTuples(dim[1] * dim[0]);

	int offset = 0;

	Point3D pt;
	vector<int> ptVec(3, 0);
	long pc = 0;
	CellG* cell;
	std::vector<float> ECMaterialsQuantityVec;

	for (int j = 0; j<dim[1]; ++j)
		for (int i = 0; i<dim[0]; ++i) {
			ptVec[0] = i;
			ptVec[1] = j;
			ptVec[2] = _pos;

			pt.x = ptVec[pointOrderVec[0]];
			pt.y = ptVec[pointOrderVec[1]];
			pt.z = ptVec[pointOrderVec[2]];

			cell = cellFieldG->get(pt);
			
			Coordinates3D<double> hexCoords = HexCoordXY(pt.x, pt.y, pt.z);
			for (int idx = 0; idx<6; ++idx) {
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

			if (!cell) {
				ECMaterialsQuantityVec = ECMaterialsField->get(pt)->ECMaterialsQuantityVec;
				for (int k = 0; k < numberOfMaterials; ++k) {
					_quantityArray->SetComponent(offset, k, (double) ECMaterialsQuantityVec[compSelVec[k]]);
				}
			}
			else {
				for (int k = 0; k < numberOfMaterials; ++k) {
					_quantityArray->SetComponent(offset, k, 0.0);
				}
			}
			++offset;
		}
}

void FieldExtractor::fillECMaterialData2DCartesian(vtk_obj_addr_int_t _ecmQuantityArrayAddr, vtk_obj_addr_int_t _cartesianCellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane, int _pos, int _compSel) {
	vtkPoints *_pointsArray = (vtkPoints *)_pointsArrayAddr;
	vtkCellArray * _cartesianCellsArray = (vtkCellArray*)_cartesianCellsArrayAddr;
	
	Field3D<CellG*> * cellFieldG = potts->getCellFieldG();
	Dim3D fieldDim = cellFieldG->getDim();

	Field3D<ECMaterialsData *> *ECMaterialsField = ecmPlugin->getECMaterialField();

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

	int numberOfMaterials;
	std::vector<int> compSelVec;
	if (_compSel < 0) {
		numberOfMaterials = (int)ecmPlugin->getNumberOfMaterials();
		for (int i = 0; i < numberOfMaterials; ++i) { compSelVec.push_back(i); }
	}
	else {
		numberOfMaterials = 1;
		compSelVec.push_back(_compSel);
	}

	vtkDoubleArray *_quantityArray = (vtkDoubleArray *)_ecmQuantityArrayAddr;
	_quantityArray->SetNumberOfComponents(numberOfMaterials);
	_quantityArray->SetNumberOfTuples(dim[1] * dim[0]);

	int offset = 0;

	Point3D pt;
	vector<int> ptVec(3, 0);
	long pc = 0;
	CellG* cell;
	std::vector<float> ECMaterialsQuantityVec;

	for (int j = 0; j<dim[1]; ++j)
		for (int i = 0; i<dim[0]; ++i) {
			ptVec[0] = i;
			ptVec[1] = j;
			ptVec[2] = _pos;

			pt.x = ptVec[pointOrderVec[0]];
			pt.y = ptVec[pointOrderVec[1]];
			pt.z = ptVec[pointOrderVec[2]];

			cell = cellFieldG->get(pt);

			Coordinates3D<double> coords(ptVec[0], ptVec[1], 0);
			for (int idx = 0; idx<4; ++idx) {
				Coordinates3D<double> cartesianVertex = cartesianVertices[idx] + coords;
				_pointsArray->InsertNextPoint(cartesianVertex.x, cartesianVertex.y, 0.0);
			}

			pc += 4;
			vtkIdType cellId = _cartesianCellsArray->InsertNextCell(4);
			_cartesianCellsArray->InsertCellPoint(pc - 4);
			_cartesianCellsArray->InsertCellPoint(pc - 3);
			_cartesianCellsArray->InsertCellPoint(pc - 2);
			_cartesianCellsArray->InsertCellPoint(pc - 1);

			if (!cell) {
				ECMaterialsQuantityVec = ECMaterialsField->get(pt)->ECMaterialsQuantityVec;
				for (int k = 0; k < numberOfMaterials; ++k) {
					_quantityArray->SetComponent(offset, k, (double)ECMaterialsQuantityVec[compSelVec[k]]);
				}
			}
			else {
				for (int k = 0; k < numberOfMaterials; ++k) {
					_quantityArray->SetComponent(offset, k, 0.0);
				}
			}
			++offset;
		}
}

void FieldExtractor::fillECMaterialFieldData3D(vtk_obj_addr_int_t _ecmQuantityArrayAddr, int _compSel) {

	Field3D<CellG*> * cellFieldG = potts->getCellFieldG();
	Dim3D fieldDim = cellFieldG->getDim();

	Field3D<ECMaterialsData *> *ECMaterialsField = ecmPlugin->getECMaterialField();

	int numberOfMaterials;
	std::vector<int> compSelVec;
	if (_compSel < 0) {
		numberOfMaterials = (int)ecmPlugin->getNumberOfMaterials();
		for (int i = 0; i < numberOfMaterials; ++i) { compSelVec.push_back(i); }
	}
	else { 
		numberOfMaterials = 1;
		compSelVec.push_back(_compSel);
	}
	
	vtkDoubleArray *_quantityArray = (vtkDoubleArray *)_ecmQuantityArrayAddr;
	_quantityArray->SetNumberOfComponents(numberOfMaterials);
	_quantityArray->SetNumberOfTuples(fieldDim.x*fieldDim.y*fieldDim.z);
	
	int offset = 0;

	Point3D pt;
	vector<int> ptVec(3, 0);
	CellG* cell;
	std::vector<float> ECMaterialsQuantityVec(numberOfMaterials);

	for (int k = 0; k<fieldDim.z; ++k)
		for (int j = 0; j<fieldDim.y; ++j)
			for (int i = 0; i<fieldDim.x; ++i) {
				pt.x = i;
				pt.y = j;
				pt.z = k;

				cell = cellFieldG->get(pt);

				if (!cell) {
					ECMaterialsQuantityVec = ECMaterialsField->get(pt)->ECMaterialsQuantityVec;
					for (int l = 0; l < numberOfMaterials; ++l) {
						_quantityArray->SetComponent(offset, l, (double) ECMaterialsQuantityVec[compSelVec[l]]);
					}
				}
				else {
					for (int l = 0; l < numberOfMaterials; ++l) {
						_quantityArray->SetComponent(offset, l, 0.0);
					}
				}
				++offset;
			}
}

void FieldExtractor::fillECMaterialDisplayField(vtk_obj_addr_int_t _colorsArrayAddr, vtk_obj_addr_int_t _quantityArrayAddr, vtk_obj_addr_int_t _colors_lutAddr) {

	vtkDoubleArray *_colorArray = (vtkDoubleArray *)_colorsArrayAddr;
	vtkDoubleArray *_quantityArray = (vtkDoubleArray *)_quantityArrayAddr;
	vtkLookupTable *_colors_lut = (vtkLookupTable *)_colors_lutAddr;
	
	int numberOfTuples = _quantityArray->GetNumberOfTuples();
	_colorArray->SetNumberOfComponents(4);
	_colorArray->SetNumberOfTuples(numberOfTuples);

	int numberOfMaterials = _quantityArray->GetNumberOfComponents();
	double thisColor[4];
	std::vector<std::vector<double> > _colors_lut_arr(numberOfMaterials, std::vector<double>(4));

	for (int colorIndex = 0; colorIndex < numberOfMaterials; ++colorIndex) {
		_colors_lut->GetTableValue(colorIndex, thisColor);
		for (int i = 0; i < 4; ++i) { _colors_lut_arr[colorIndex][i] = thisColor[i]; }
	}

	double * thisQuantityTuple;
	for (int tupleIndex = 0; tupleIndex < numberOfTuples; ++tupleIndex ) {
		thisQuantityTuple = _quantityArray->GetTuple(tupleIndex);
		double thisColorTuple[4] = { 0.0, 0.0, 0.0, 0.0 };
		for (int materialIndex = 0; materialIndex < numberOfMaterials; ++materialIndex) {
			for (int colorIndex = 0; colorIndex < 4; ++colorIndex) {
				thisColorTuple[colorIndex] += thisQuantityTuple[materialIndex]*_colors_lut_arr[materialIndex][colorIndex];
			}
		}
		_colorArray->SetTuple(tupleIndex, thisColorTuple);
	}

}

void FieldExtractor::setVtkObj(void * _vtkObj){
	cerr<<"INSIDE setVtkObj"<<endl;
}

void FieldExtractor::setVtkObjInt(long _vtkObjAddr){
	void * vPtr=(void*)_vtkObjAddr;
	cerr<<"GOT THIS VOID ADDR "<<vPtr<<endl;
	vtkIntArray * arrayPtr=(vtkIntArray *)vPtr;
	arrayPtr->SetName("INTEGER ARRAY");
	cerr<<"THIS IS NAME OF THE ARRAY="<<arrayPtr->GetName()<<endl;
}

vtkIntArray * FieldExtractor::produceVtkIntArray(){
	vtkIntArray * vtkIntArrayObj=vtkIntArray::New();
	return vtkIntArrayObj;
}

int * FieldExtractor::produceArray(int _size){
	return new int[_size];
}
