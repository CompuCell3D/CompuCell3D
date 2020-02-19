

#include <iostream>
#include <sstream>
#include <CompuCell3D/Field3D/Dim3D.h>

#include <Utils/Coordinates3D.h>
#include <algorithm>
#include <cmath>
#include <set>

#include <vtkPythonUtil.h>
#include <vtkDoubleArray.h>
#include <vtkLookupTable.h>

using namespace std;
using namespace CompuCell3D;


#include "FieldExtractorBase.h"


FieldExtractorBase::FieldExtractorBase()
{
	
	double sqrt_3_3=sqrt(3.0)/3.0;
	hexagonVertices.push_back(Coordinates3D<double>(0, sqrt_3_3, 0.0));
	hexagonVertices.push_back(Coordinates3D<double>(0.5 , 0.5*sqrt_3_3, 0.0));
	hexagonVertices.push_back(Coordinates3D<double>(0.5, -0.5*sqrt_3_3, 0.0));
	hexagonVertices.push_back(Coordinates3D<double>(0. , -sqrt_3_3, 0.0));
	hexagonVertices.push_back(Coordinates3D<double>(-0.5 , -0.5*sqrt_3_3, 0.0));
	hexagonVertices.push_back(Coordinates3D<double>(-0.5, 0.5*sqrt_3_3, 0.0));
    
    cartesianVertices.push_back(Coordinates3D<double>(0.0, 0.0, 0.0));
    cartesianVertices.push_back(Coordinates3D<double>(0.0, 1.0, 0.0));
    cartesianVertices.push_back(Coordinates3D<double>(1.0, 1.0, 0.0));
    cartesianVertices.push_back(Coordinates3D<double>(1.0, 0.0, 0.0));


}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldExtractorBase::~FieldExtractorBase(){

}




////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<int> FieldExtractorBase::pointOrder(std::string _plane){
	for (int i = 0  ; i <_plane.size() ; ++i){
		_plane[i]=tolower(_plane[i]);
	}

	std::vector<int> order(3,0);
	order[0] =0;
	order[1] =1;
	order[2] =2;
	if (_plane == "xy"){
		order[0] =0;
		order[1] =1;
		order[2] =2;            
	}
	else if (_plane == "xz"){
		order[0] =0;
		order[1] =2;
		order[2] =1;            


	}
	else if( _plane == "yz"){ 
		order[0] =2;
		order[1] =0;
		order[2] =1;            



	}
	return order;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<int> FieldExtractorBase::dimOrder(std::string _plane){
	for (int i = 0  ; i <_plane.size() ; ++i){
		_plane[i]=tolower(_plane[i]);
	}

	std::vector<int> order(3,0);
	order[0] =0;
	order[1] =1;
	order[2] =2;
	if (_plane == "xy"){
		order[0] =0;
		order[1] =1;
		order[2] =2;            
	}
	else if (_plane == "xz"){
		order[0] =0;
		order[1] =2;
		order[2] =1;            


	}
	else if( _plane == "yz"){ 
		order[0] =1;
		order[1] =2;
		order[2] =0;            



	}
	return order;
}

Coordinates3D<double> FieldExtractorBase::HexCoordXY(unsigned int x,unsigned int y,unsigned int z){
    //coppied from BoundaryStrategy.cpp HexCoord fcn
   if((z%3)==1){//odd z e.g. z=1

      if(y%2)
         return Coordinates3D<double>(x+0.5 , sqrt(3.0)/2.0*(y+2.0/6.0), z*sqrt(6.0)/3.0);
      else//even
         return Coordinates3D<double>( x ,  sqrt(3.0)/2.0*(y+2.0/6.0) , z*sqrt(6.0)/3.0);


   }else if((z%3)==2){ //e.g. z=2


      if(y%2)
         return Coordinates3D<double>(x+0.5 , sqrt(3.0)/2.0*(y-2.0/6.0), z*sqrt(6.0)/3.0);
      else//even
         return Coordinates3D<double>( x ,  sqrt(3.0)/2.0*(y-2.0/6.0) , z*sqrt(6.0)/3.0);




   }
   else{//z divible by 3 - includes z=0
      if(y%2)
         return Coordinates3D<double>(x , sqrt(3.0)/2.0*y, z*sqrt(6.0)/3.0);
      else//even
         return Coordinates3D<double>( x+0.5 ,  sqrt(3.0)/2.0*y , z*sqrt(6.0)/3.0);
   }


}

void FieldExtractorBase::fillCellFieldData2D(vtk_obj_addr_int_t _cellTypeArrayAddr , std::string _plane ,  int _pos){}

void FieldExtractorBase::fillNCMaterialDisplayField(vtk_obj_addr_int_t _colorsArrayAddr, vtk_obj_addr_int_t _quantityArrayAddr, vtk_obj_addr_int_t _colors_lutAddr) {

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
	for (int tupleIndex = 0; tupleIndex < numberOfTuples; ++tupleIndex) {
		thisQuantityTuple = _quantityArray->GetTuple(tupleIndex);
		double thisColorTuple[4] = { 0.0, 0.0, 0.0, 0.0 };
		for (int materialIndex = 0; materialIndex < numberOfMaterials; ++materialIndex) {
			for (int colorIndex = 0; colorIndex < 4; ++colorIndex) {
				thisColorTuple[colorIndex] += thisQuantityTuple[materialIndex] * _colors_lut_arr[materialIndex][colorIndex];
			}
		}
		_colorArray->SetTuple(tupleIndex, thisColorTuple);
	}

}