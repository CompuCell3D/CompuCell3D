

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

using namespace std;
using namespace CompuCell3D;


#include "FieldExtractorCML.h"


FieldExtractorCML::FieldExtractorCML():lds(0),zDimFactor(0),yDimFactor(0)
{

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldExtractorCML::~FieldExtractorCML(){

}

void FieldExtractorCML::setSimulationData(long _structuredPointsAddr){

	lds=(vtkStructuredPoints *)_structuredPointsAddr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Dim3D FieldExtractorCML::getFieldDim(){
	return fieldDim;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FieldExtractorCML::setFieldDim(Dim3D _dim){
	fieldDim=_dim;
	zDimFactor=fieldDim.x*fieldDim.y;
	yDimFactor=fieldDim.x;
}

long FieldExtractorCML::pointIndex(short _x,short _y,short _z){
	return zDimFactor*_z+yDimFactor*_y+_x;
}

long FieldExtractorCML::indexPoint3D(Point3D _pt){
	return zDimFactor*_pt.z+yDimFactor*_pt.y+_pt.x;
}

void FieldExtractorCML::fillCentroidData2D(long _pointArrayAddr ,long _linesArrayAddr, std::string _plane ,  int _pos){

}

void FieldExtractorCML::fillCellFieldData2D(long _cellTypeArrayAddr, std::string _plane, int _pos){

	//cerr<<" \n\n\n THIS IS fillCellFieldData2D\n\n\n\n"<<endl;
	vtkIntArray *_cellTypeArray=(vtkIntArray *)_cellTypeArrayAddr;

	// get cell type array from vtk structured points
	//cerr<<"lds="<<lds<<endl;
	//cerr<<lds->GetPointData()->GetArray("CellType")<<endl;
	//cerr<<"STRUCTURED DATA POINTS="<<lds<<endl;
	
	//lds->Print(cerr);

	//vtkPointData * pointDataPtr=lds->GetPointData();
	//cerr<<"pointDataPtr="<<pointDataPtr<<endl;
	//vtkDataArray * dataArrayPtr=lds->GetPointData()->GetArray("CellType");
	//cerr<<"dataArrayPtr="<<dataArrayPtr<<endl;
	vtkCharArray *typeArrayRead=(vtkCharArray *)lds->GetPointData()->GetArray("CellType");
	//cerr<<"typeArrayRead="<<typeArrayRead<<endl;


	//typeArrayRead->Print(cerr);

	//->GetArray("CellType");

	//cerr<<"fieldDim.x="<<fieldDim.x<<endl;
	//cerr<<"fieldDim.y="<<fieldDim.y<<endl;
	//cerr<<"fieldDim.z="<<fieldDim.z<<endl;

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

	int type;
	//long index;
	//when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
	for(int j =0 ; j<dim[1]+1 ; ++j)
		for(int i =0 ; i<dim[0]+1 ; ++i){
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

			if(i>=dim[0] ||j>=dim[1] ){
				_cellTypeArray->SetValue(offset, 0);
			}else{
				_cellTypeArray->SetValue(offset, typeArrayRead->GetValue(indexPoint3D(pt)));
			}
			++offset;
		}
}

void FieldExtractorCML::fillCellFieldData2DHex(long _cellTypeArrayAddr,long _hexCellsArrayAddr ,long _pointsArrayAddr, std::string _plane ,  int _pos){
	vtkIntArray *_cellTypeArray=(vtkIntArray *)_cellTypeArrayAddr;
	vtkPoints *_pointsArray=(vtkPoints *)_pointsArrayAddr;
	vtkCellArray * _hexCellsArray=(vtkCellArray*)_hexCellsArrayAddr;

	vtkCharArray *typeArrayRead=(vtkCharArray *)lds->GetPointData()->GetArray("CellType");

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

	char cellType;

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

			cellType=typeArrayRead->GetValue(indexPoint3D(pt));

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

			_cellTypeArray->InsertNextValue(cellType);

			++offset;
		}
}
void FieldExtractorCML::fillBorder2D(const char* arrayName, long _pointArrayAddr ,long _linesArrayAddr, std::string _plane ,  int _pos){

//	vtkLongArray *idArray=(vtkLongArray *)lds->GetPointData()->GetArray("CellId");
	vtkLongArray *idArray=(vtkLongArray *)lds->GetPointData()->GetArray(arrayName);

	vtkPoints *points = (vtkPoints *)_pointArrayAddr;
	vtkCellArray * lines=(vtkCellArray *)_linesArrayAddr; 

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

	long idxPt;

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

			idxPt=indexPoint3D(pt);

			if(i > 0 && j < dim[1] ){
				ptNVec[0]=i-1;
				ptNVec[1]=j;
				ptNVec[2]=_pos;
				ptN.x=ptNVec[pointOrderVec[0]];
				ptN.y=ptNVec[pointOrderVec[1]];
				ptN.z=ptNVec[pointOrderVec[2]];
				if(idArray->GetValue(idxPt)!=idArray->GetValue(indexPoint3D(ptN))){

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
				if(idArray->GetValue(idxPt)!=idArray->GetValue(indexPoint3D(ptN))){

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
				if(ptNVec[0]>=dim[0] || idArray->GetValue(idxPt)!=idArray->GetValue(indexPoint3D(ptN))){

					points->InsertNextPoint(i+1,j,0);
					points->InsertNextPoint(i+1,j+1,0);
					pc+=2;
					lines->InsertNextCell(2);
					lines->InsertCellPoint(pc-2);
					lines->InsertCellPoint(pc-1);
				}
			}
			if( i < dim[0] && j < dim[1]  ) {
				ptNVec[0]=i;
				ptNVec[1]=j+1;
				ptNVec[2]=_pos;
				ptN.x=ptNVec[pointOrderVec[0]];
				ptN.y=ptNVec[pointOrderVec[1]];
				ptN.z=ptNVec[pointOrderVec[2]];
				if(ptNVec[1]>=dim[1] || idArray->GetValue(idxPt)!=idArray->GetValue(indexPoint3D(ptN))){

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
void FieldExtractorCML::fillBorderData2D(long _pointArrayAddr ,long _linesArrayAddr, std::string _plane ,  int _pos)
{
	fillBorder2D("CellId", _pointArrayAddr ,_linesArrayAddr, _plane , _pos);
}
void FieldExtractorCML::fillClusterBorderData2D(long _pointArrayAddr ,long _linesArrayAddr, std::string _plane ,  int _pos)
{
	fillBorder2D("ClusterId", _pointArrayAddr ,_linesArrayAddr, _plane , _pos);
}

void FieldExtractorCML::fillBorder2DHex(const char* arrayName, long _pointArrayAddr ,long _linesArrayAddr, std::string _plane ,  int _pos){

	vtkPoints *points = (vtkPoints *)_pointArrayAddr;
	vtkCellArray * lines=(vtkCellArray *)_linesArrayAddr; 

//	vtkLongArray *idArray=(vtkLongArray *)lds->GetPointData()->GetArray("CellId");
	vtkLongArray *idArray=(vtkLongArray *)lds->GetPointData()->GetArray(arrayName);

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
	long idxPt;

	for(int i=0; i <dim[0]; ++i)
		for(int j=0; j <dim[1]; ++j){
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];
			Coordinates3D<double> hexCoords=HexCoordXY(pt.x,pt.y,pt.z);
			idxPt=indexPoint3D(pt);

			if(pt.y%2){ //y_odd
				if(pt.x-1>=0){
					ptN.x=pt.x-1;
					ptN.y=pt.y;
					ptN.z=pt.z;
					if(idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN)) ){

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
					if(idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN)) ){

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
					if(idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN)) ){

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
					if(idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN)) ){

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
					if(idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN)) ){

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
					if(idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN)) ){

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
					if(idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN)) ){
						//if(cellFieldG->get(pt) != cellFieldG->get(ptN)){
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
					if(idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN)) ){

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
					if(idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN)) ){

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
					if(idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN)) ){

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
					if(idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN)) ){

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
					if(idArray->GetValue(idxPt) != idArray->GetValue(indexPoint3D(ptN)) ){

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
void FieldExtractorCML::fillBorderData2DHex(long _pointArrayAddr ,long _linesArrayAddr, std::string _plane ,  int _pos)
{
	fillBorder2DHex("CellId", _pointArrayAddr ,_linesArrayAddr, _plane , _pos);
}
void FieldExtractorCML::fillClusterBorderData2DHex(long _pointArrayAddr ,long _linesArrayAddr, std::string _plane ,  int _pos)
{
	fillBorder2DHex("ClusterId", _pointArrayAddr ,_linesArrayAddr, _plane , _pos);
}

bool FieldExtractorCML::fillConFieldData2DHex(long _conArrayAddr,long _hexCellsArrayAddr ,long _pointsArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos){

	vtkDoubleArray *conArrayRead=(vtkDoubleArray *)lds->GetPointData()->GetArray(_conFieldName.c_str());

	if (!conArrayRead)
		return false;

	vtkDoubleArray *conArray=(vtkDoubleArray *)_conArrayAddr;

	vtkCellArray * _hexCellsArray=(vtkCellArray*)_hexCellsArrayAddr;

	vtkPoints *_pointsArray=(vtkPoints *)_pointsArrayAddr;

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
				con = conArrayRead->GetValue(indexPoint3D(pt));
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

bool FieldExtractorCML::fillScalarFieldData2DHex(long _conArrayAddr,long _hexCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){

	return fillConFieldData2DHex(_conArrayAddr,_hexCellsArrayAddr ,_pointsArrayAddr,_conFieldName, _plane ,   _pos)	;
}

bool FieldExtractorCML::fillScalarFieldCellLevelData2DHex(long _conArrayAddr,long _hexCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){

	return fillConFieldData2DHex(_conArrayAddr,_hexCellsArrayAddr ,_pointsArrayAddr,_conFieldName, _plane ,   _pos)	;
}

bool FieldExtractorCML::fillConFieldData2D(long _conArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos){    
	vtkDoubleArray *conArrayRead=(vtkDoubleArray *)lds->GetPointData()->GetArray(_conFieldName.c_str());

	if (!conArrayRead)
		return false;

	vtkDoubleArray *conArray=(vtkDoubleArray *)_conArrayAddr;

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
				con = conArrayRead->GetValue(indexPoint3D(pt));
			}
			conArray->SetValue(offset, con);
			++offset;
		}
		return true;
}

bool FieldExtractorCML::fillScalarFieldData2D(long _conArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos){

	return fillConFieldData2D( _conArrayAddr, _conFieldName,  _plane ,   _pos);
}

bool FieldExtractorCML::fillScalarFieldData2DCartesian(long _conArrayAddr,long _cartesianCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){
    return fillConFieldData2DCartesian(_conArrayAddr,_cartesianCellsArrayAddr , _pointsArrayAddr , _conFieldName , _plane ,_pos);
}

bool FieldExtractorCML::fillConFieldData2DCartesian(long _conArrayAddr,long _cartesianCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){
    vtkDoubleArray *conArrayRead=(vtkDoubleArray *)lds->GetPointData()->GetArray(_conFieldName.c_str());

	if (!conArrayRead)
		return false;

	vtkDoubleArray *conArray=(vtkDoubleArray *)_conArrayAddr;

	vtkCellArray * _cartesianCellsArray=(vtkCellArray*)_cartesianCellsArrayAddr;

	vtkPoints *_pointsArray=(vtkPoints *)_pointsArrayAddr;

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
				con = conArrayRead->GetValue(indexPoint3D(pt));
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


bool FieldExtractorCML::fillScalarFieldCellLevelData2D(long _conArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos){

	return fillConFieldData2D( _conArrayAddr, _conFieldName,  _plane ,   _pos);
}

bool FieldExtractorCML::fillScalarFieldCellLevelData2DCartesian(long _conArrayAddr,long _cartesianCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){
    return fillConFieldData2DCartesian(_conArrayAddr,_cartesianCellsArrayAddr , _pointsArrayAddr , _conFieldName , _plane ,_pos);
}


bool FieldExtractorCML::fillVectorFieldData2D(long _pointsArrayIntAddr,long _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos){
	vtkFloatArray * vectorArray=(vtkFloatArray *)_vectorArrayIntAddr;
	vtkPoints *pointsArray=(vtkPoints *)_pointsArrayIntAddr;

	vtkFloatArray *vecArrayRead = (vtkFloatArray *)lds->GetPointData()->GetArray(_fieldName.c_str());

	if (!vecArrayRead)
		return false;

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

	float  vecTmp[3] ;
	double  vecTmpCoord[3] ;
	//double con;

	int offset=0;

	for(int j =0 ; j<dim[1] ; ++j)
		for(int i =0 ; i<dim[0] ; ++i){
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

			vecArrayRead->GetTuple(indexPoint3D(pt),vecTmpCoord);

			if(vecTmpCoord[0]!=0.0 || vecTmpCoord[1]!=0.0 || vecTmpCoord[2]!=0.0){
				pointsArray->InsertPoint(offset,ptVec[0],ptVec[1],0);
				vectorArray->InsertTuple3(offset,vecTmpCoord[pointOrderVec[0]],vecTmpCoord[pointOrderVec[1]],0);
				++offset;
			}
		}
		return true;
}

bool FieldExtractorCML::fillVectorFieldData2DHex(long _pointsArrayIntAddr,long _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos){
	vtkFloatArray * vectorArray=(vtkFloatArray *)_vectorArrayIntAddr;
	vtkPoints *pointsArray=(vtkPoints *)_pointsArrayIntAddr;

	vtkFloatArray *vecArrayRead = (vtkFloatArray *)lds->GetPointData()->GetArray(_fieldName.c_str());

	if (!vecArrayRead)
		return false;

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

	float  vecTmp[3] ;
	double  vecTmpCoord[3] ;

	int offset=0;

	for(int j =0 ; j<dim[1] ; ++j)
		for(int i =0 ; i<dim[0] ; ++i){
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

			vecArrayRead->GetTuple(indexPoint3D(pt),vecTmpCoord);

			if(vecTmpCoord[0]!=0.0 || vecTmpCoord[1]!=0.0 || vecTmpCoord[2]!=0.0){

				Coordinates3D<double> hexCoords=HexCoordXY(pt.x,pt.y,pt.z);
				pointsArray->InsertPoint(offset, hexCoords.x,hexCoords.y,0.0);

				vectorArray->InsertTuple3(offset,vecTmpCoord[pointOrderVec[0]],vecTmpCoord[pointOrderVec[1]],0);
				++offset;
			}
		}
		return true;
}

bool FieldExtractorCML::fillVectorFieldData3D(long _pointsArrayIntAddr,long _vectorArrayIntAddr,std::string _fieldName){

	vtkFloatArray * vectorArray=(vtkFloatArray *)_vectorArrayIntAddr;
	vtkPoints *pointsArray=(vtkPoints *)_pointsArrayIntAddr;

	vtkFloatArray *vecArrayRead = (vtkFloatArray *)lds->GetPointData()->GetArray(_fieldName.c_str());

	if (!vecArrayRead)
		return false;

	Point3D pt;
	vector<int> ptVec(3,0);

	double vecTmp[3];

	int offset=0;
	for(pt.z =0 ; pt.z<fieldDim.z ; ++pt.z)	
		for(pt.y =0 ; pt.y<fieldDim.y ; ++pt.y)
			for(pt.x =0 ; pt.x<fieldDim.x ; ++pt.x){

				vecArrayRead->GetTuple(indexPoint3D(pt),vecTmp);
				if(vecTmp[0]!=0.0 || vecTmp[1]!=0.0 || vecTmp[2]!=0.0){
					pointsArray->InsertPoint(offset,pt.x,pt.y,pt.z);
					vectorArray->InsertTuple3(offset,vecTmp[0],vecTmp[1],vecTmp[2]);
					++offset;
				}
			}
			return true;
}

bool FieldExtractorCML::fillVectorFieldCellLevelData2D(long _pointsArrayIntAddr,long _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos){
	vtkFloatArray * vectorArray=(vtkFloatArray *)_vectorArrayIntAddr;
	vtkPoints *pointsArray=(vtkPoints *)_pointsArrayIntAddr;

	set<long> visitedCells;

	vtkLongArray *idArray=(vtkLongArray *)lds->GetPointData()->GetArray("CellId");

	vtkFloatArray *vecArrayRead = (vtkFloatArray *)lds->GetPointData()->GetArray(_fieldName.c_str());

	if (!vecArrayRead)
		return false;

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

	long cellId;
	long idx;
	Coordinates3D<float> vecTmp;
	double vecTmpCoord[3] ;

	int offset=0;

	for(int j =0 ; j<dim[1] ; ++j)
		for(int i =0 ; i<dim[0] ; ++i){
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

			idx=indexPoint3D(pt);

			cellId=idArray->GetValue(idx);

			if(cellId){
				//check if this cell is in the set of visited Cells
				if(visitedCells.find(cellId)!=visitedCells.end()){
					continue; //cell have been visited 
				}else{
					//this is first time we visit given cell

					vecArrayRead->GetTuple(idx,vecTmpCoord);						

					if(vecTmpCoord[0]!=0.0 || vecTmpCoord[1]!=0.0 || vecTmpCoord[2]!=0.0){


						pointsArray->InsertPoint(offset,ptVec[0],ptVec[1],0);						
						vectorArray->InsertTuple3(offset,vecTmpCoord[pointOrderVec[0]],vecTmpCoord[pointOrderVec[1]],0);
						++offset;
					}
					visitedCells.insert(cellId);
				}
			}
		}
		return true;
}

bool FieldExtractorCML::fillVectorFieldCellLevelData2DHex(long _pointsArrayIntAddr,long _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos){

	vtkFloatArray * vectorArray=(vtkFloatArray *)_vectorArrayIntAddr;
	vtkPoints *pointsArray=(vtkPoints *)_pointsArrayIntAddr;

	set<long> visitedCells;

	vtkLongArray *idArray=(vtkLongArray *)lds->GetPointData()->GetArray("CellId");

	vtkFloatArray *vecArrayRead = (vtkFloatArray *)lds->GetPointData()->GetArray(_fieldName.c_str());

	if (!vecArrayRead)
		return false;

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
	long cellId;
	long idx;
	Coordinates3D<float> vecTmp;
	double vecTmpCoord[3] ;

	int offset=0;

	for(int j =0 ; j<dim[1] ; ++j)
		for(int i =0 ; i<dim[0] ; ++i){
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

			idx=indexPoint3D(pt);

			cellId=idArray->GetValue(idx);

			if(cellId){
				//check if this cell is in the set of visited Cells
				if(visitedCells.find(cellId)!=visitedCells.end()){
					continue; //cell have been visited 
				}else{
					//this is first time we visit given cell

					vecArrayRead->GetTuple(idx,vecTmpCoord);						

					if(vecTmpCoord[0]!=0.0 || vecTmpCoord[1]!=0.0 || vecTmpCoord[2]!=0.0){

						Coordinates3D<double> hexCoords=HexCoordXY(pt.x,pt.y,pt.z);
						pointsArray->InsertPoint(offset, hexCoords.x,hexCoords.y,0.0);

						vectorArray->InsertTuple3(offset,vecTmpCoord[pointOrderVec[0]],vecTmpCoord[pointOrderVec[1]],0);
						++offset;
					}
					visitedCells.insert(cellId);
				}
			}
		}
		return true;
}

bool FieldExtractorCML::fillVectorFieldCellLevelData3D(long _pointsArrayIntAddr,long _vectorArrayIntAddr,std::string _fieldName){
	vtkFloatArray * vectorArray=(vtkFloatArray *)_vectorArrayIntAddr;
	vtkPoints *pointsArray=(vtkPoints *)_pointsArrayIntAddr;

	set<long> visitedCells;

	vtkLongArray *idArray=(vtkLongArray *)lds->GetPointData()->GetArray("CellId");

	vtkFloatArray *vecArrayRead = (vtkFloatArray *)lds->GetPointData()->GetArray(_fieldName.c_str());

	if (!vecArrayRead)
		return false;

	Point3D pt;
	vector<int> ptVec(3,0);
	long cellId;
	long idx;

	double vecTmp[3];

	int offset=0;
	for(pt.z =0 ; pt.z<fieldDim.z ; ++pt.z)	
		for(pt.y =0 ; pt.y<fieldDim.y ; ++pt.y)
			for(pt.x =0 ; pt.x<fieldDim.x ; ++pt.x){

				idx=indexPoint3D(pt);

				cellId=idArray->GetValue(idx);

				if(cellId){
					//check if this cell is in the set of visited Cells
					if(visitedCells.find(cellId)!=visitedCells.end()){
						continue; //cell have been visited 
					}else{
						//this is first time we visit given cell
						vecArrayRead->GetTuple(idx,vecTmp);	
						if(vecTmp[0]!=0.0 || vecTmp[1]!=0.0 || vecTmp[2]!=0.0){


							pointsArray->InsertPoint(offset,pt.x,pt.y,pt.z);
							vectorArray->InsertTuple3(offset,vecTmp[0],vecTmp[1],vecTmp[2]);
							++offset;
						}
						visitedCells.insert(cellId);
					}
				}
			}
			return true;
}

//vector<int> FieldExtractorCML::fillCellFieldData3D(long _cellTypeArrayAddr){
vector<int> FieldExtractorCML::fillCellFieldData3D(long _cellTypeArrayAddr, long _cellIdArrayAddr){
	set<int> usedCellTypes;

	vtkIntArray *cellTypeArray = (vtkIntArray *)_cellTypeArrayAddr;
	vtkLongArray *cellIdArray = (vtkLongArray *)_cellIdArrayAddr;

	vtkCharArray *typeArrayRead = (vtkCharArray *)lds->GetPointData()->GetArray("CellType");
	vtkLongArray *idArrayRead = (vtkLongArray *)lds->GetPointData()->GetArray("CellId");

	cellTypeArray->SetNumberOfValues((fieldDim.x+2)*(fieldDim.y+2)*(fieldDim.z+2));
    cellIdArray->SetNumberOfValues((fieldDim.x+2)*(fieldDim.y+2)*(fieldDim.z+2));

	Point3D pt;
	int type;
    int id;
    long idxPt;
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
					idxPt=indexPoint3D(pt);
					type=typeArrayRead->GetValue(idxPt);
                    id=idArrayRead->GetValue(idxPt);

					if(type!=0)
						usedCellTypes.insert(type);

					cellTypeArray->InsertValue(offset, type);
                    cellIdArray->InsertValue(offset, id);

					++offset;
				}
			}
			return vector<int>(usedCellTypes.begin(),usedCellTypes.end());
}

bool FieldExtractorCML::fillConFieldData3D(long _conArrayAddr ,long _cellTypeArrayAddr, std::string _conFieldName,std::vector<int> * _typesInvisibeVec){
	vtkDoubleArray *conArray=(vtkDoubleArray *)_conArrayAddr;
	vtkIntArray *cellTypeArray=(vtkIntArray *)_cellTypeArrayAddr;

	vtkCharArray *typeArrayRead=(vtkCharArray *)lds->GetPointData()->GetArray("CellType");

	vtkDoubleArray *conArrayRead=(vtkDoubleArray *)lds->GetPointData()->GetArray(_conFieldName.c_str());

	if (!conArrayRead)
		return false;

	conArray->SetNumberOfValues((fieldDim.x+2)*(fieldDim.y+2)*(fieldDim.z+2));
	cellTypeArray->SetNumberOfValues((fieldDim.x+2)*(fieldDim.y+2)*(fieldDim.z+2));

	set<int> invisibleTypeSet(_typesInvisibeVec->begin(),_typesInvisibeVec->end());

	//for (set<int>::iterator sitr=invisibleTypeSet.begin();sitr!=invisibleTypeSet.end();++sitr){
	//	cerr<<"invisible type="<<*sitr<<endl;
	//}

	Point3D pt;
	long idxPt;
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

					idxPt=indexPoint3D(pt);
					con=conArrayRead->GetValue(idxPt);
					type=typeArrayRead->GetValue(idxPt);

					if(type && invisibleTypeSet.find(type)!=invisibleTypeSet.end()){
						type=0;
					}

					conArray->InsertValue(offset, con);
					cellTypeArray->InsertValue(offset, type);
					++offset;
				}
			}
			return true;
}

bool FieldExtractorCML::fillScalarFieldData3D(long _conArrayAddr ,long _cellTypeArrayAddr, std::string _conFieldName,std::vector<int> * _typesInvisibeVec){

	return fillConFieldData3D(_conArrayAddr ,_cellTypeArrayAddr, _conFieldName, _typesInvisibeVec);
}

bool FieldExtractorCML::fillScalarFieldCellLevelData3D(long _conArrayAddr ,long _cellTypeArrayAddr, std::string _conFieldName,std::vector<int> * _typesInvisibeVec){

	return fillConFieldData3D(_conArrayAddr ,_cellTypeArrayAddr, _conFieldName, _typesInvisibeVec);
}


bool FieldExtractorCML::readVtkStructuredPointsData(long _structuredPointsReaderAddr){
    vtkStructuredPointsReader * reader=(vtkStructuredPointsReader *)_structuredPointsReaderAddr;
    reader->Update();

    return true;
}
