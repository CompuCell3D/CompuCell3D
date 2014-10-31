    
//#include "CellGraphicsData.h"
#include <iostream>
// #include <CompuCell3D/Simulator.h>
#include <CA/CAManager.h>
#include <CA/CACellStack.h>
#include <CA/CACell.h>
#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <Utils/Coordinates3D.h>
#include <vtkIntArray.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkType.h>
#include <algorithm>
#include <cmath>

#include <vtkPythonUtil.h>

using namespace std;
using namespace CompuCell3D;


#include "CAFieldExtractor.h"

CAFieldExtractor::CAFieldExtractor():caManager(0)
{
	cartesianVertices.push_back(Coordinates3D<double>(0.0, 0.0, 0.0));
    cartesianVertices.push_back(Coordinates3D<double>(0.0, 1.0, 0.0));
    cartesianVertices.push_back(Coordinates3D<double>(1.0, 1.0, 0.0));
    cartesianVertices.push_back(Coordinates3D<double>(1.0, 0.0, 0.0));


}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CAFieldExtractor::~CAFieldExtractor(){

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CAFieldExtractor::init(CAManager * _caManager){
    caManager=_caManager;
}

void* CAFieldExtractor::unmangleSWIGVktPtr(std::string _swigStyleVtkPtr){
	
	void *ptr;
	char typeCheck[128];
	int i;
	if (_swigStyleVtkPtr.size()<128){
		i = sscanf(_swigStyleVtkPtr.c_str(),"_%lx_%s",(long *)&ptr,typeCheck);
		return ptr;
	}
	else{
		return 0;
	}
	
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
long CAFieldExtractor::unmangleSWIGVktPtrAsLong(std::string _swigStyleVtkPtr){
	return (long)unmangleSWIGVktPtr( _swigStyleVtkPtr);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CAFieldExtractor::extractCellField(){
cerr<<" INSIDE CAFieldExtractor::extractCellField"<<endl;

}

// void FieldExtractor::extractCellField(){
	// //cerr<<"EXTRACTING CELL FIELD"<<endl;
	// Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	// Dim3D fieldDim=cellFieldG->getDim();
	// Point3D pt;
	// // cerr<< "FIeld Extractor cell field fieldDim="<<fieldDim<<endl;
	// CellGraphicsData gd;
	// CellG *cell;

	// for(pt.x =0 ; pt.x < fieldDim.x ; ++pt.x)
		// for(pt.y =0 ; pt.y < fieldDim.y ; ++pt.y)
			// for(pt.z =0 ; pt.z < fieldDim.z ; ++pt.z){
				// cell=cellFieldG->get(pt);
				// if(!cell){
					// gd.type=0;
					// gd.id=0;
				// }else{
					// gd.type=cell->type;
					// gd.id=cell->id;
				// }
				// fsPtr->field3DGraphicsData[pt.x][pt.y][pt.z]=gd;
			// }
// }

void CAFieldExtractor::fillCellFieldData2D(long _cellTypeArrayAddr , long _centroidPointsAddr, long _scaleRadiusArrayAddr, std::string _plane, int _pos){

   vtkIntArray *_cellTypeArray = (vtkIntArray *)_cellTypeArrayAddr;
   vtkPoints * _centroidPoints = (vtkPoints *) _centroidPointsAddr;
   vtkFloatArray * _scaleRadiusArray= (vtkFloatArray *) _scaleRadiusArrayAddr;

   cerr<<"\t\t\t\t\t INSIDE CAFieldExtractor::fillCellFieldData2D"<<endl;
   cerr<<"_cellTypeArrayAddr="<<_cellTypeArrayAddr<<endl;
   cerr<<"_plane="<<_plane<<endl;
   cerr<<"_pos="<<_pos<<endl;


	Field3D<CACellStack *> * cellField  = caManager->getCellFieldS();
	Dim3D fieldDim=cellField->getDim();

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
	CACellStack* cellStack;
	int type;

	for(int j =0 ; j<dim[1]+1 ; ++j)
		for(int i =0 ; i<dim[0]+1 ; ++i){
			ptVec[0]=i;
			ptVec[1]=j;
			ptVec[2]=_pos;

			pt.x=ptVec[pointOrderVec[0]];
			pt.y=ptVec[pointOrderVec[1]];
			pt.z=ptVec[pointOrderVec[2]];

            cellStack = cellField->get(pt);
			if (cellStack){
				int size = cellStack->getFillLevel();
				for (int idx  = 0 ; idx < size ; ++idx){
					CACell * cell = cellStack->getCellByIdx(idx);
					_centroidPoints -> InsertNextPoint(i+idx/(float)size,j+idx/(float)size,0.0);
					_cellTypeArray->InsertNextValue(cell->type);
					_scaleRadiusArray ->InsertNextValue(0.5); //or now we use size 0.5 for all glyphs this might change 

				}

			}
		}

}

void CAFieldExtractor::fillCellFieldData3D(long _cellTypeArrayAddr , long _centroidPointsAddr, long _scaleRadiusArrayAddr){

   vtkIntArray *_cellTypeArray = (vtkIntArray *)_cellTypeArrayAddr;
   vtkPoints * _centroidPoints = (vtkPoints *) _centroidPointsAddr;
   vtkFloatArray * _scaleRadiusArray = (vtkFloatArray *) _scaleRadiusArrayAddr;

   cerr<<"INSIDE CAFieldExtractor::fillCellFieldData3D"<<endl;

	Field3D<CACellStack *> * cellField  = caManager->getCellFieldS();
	Dim3D fieldDim=cellField->getDim();

	Point3D pt;
	CACellStack* cellStack;
	int type;

	for(pt.x = 0 ; pt.x < fieldDim.x ; ++pt.x)
		for(pt.y = 0 ; pt.y < fieldDim.y ; ++pt.y)
			for(pt.z = 0 ; pt.z < fieldDim.z ; ++pt.z){
		
				cellStack = cellField->get(pt);
				if (cellStack){
					int size = cellStack->getFillLevel();
					for (int idx  = 0 ; idx < size ; ++idx){
						CACell * cell = cellStack->getCellByIdx(idx);
						_centroidPoints -> InsertNextPoint(pt.x+idx/(float)size,pt.y+idx/(float)size,pt.z+idx/(float)size);
						_cellTypeArray->InsertNextValue(cell->type);
						_scaleRadiusArray ->InsertNextValue(0.5); //or now we use size 0.5 for all glyphs this might change 

				}

			}
		}

}

bool CAFieldExtractor::fillScalarFieldData2DCartesian(long _conArrayAddr,long _cartesianCellsArrayAddr ,long _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos){
	vtkDoubleArray *conArray=(vtkDoubleArray *)_conArrayAddr;
	vtkCellArray * _cartesianCellsArray=(vtkCellArray*)_cartesianCellsArrayAddr;
	vtkPoints *_pointsArray=(vtkPoints *)_pointsArrayAddr;

	Field3D<float>* conFieldPtr=caManager->getConcentrationField(_conFieldName);
		

	cerr<<"conFieldPtr="<<conFieldPtr<<endl;

	if(!conFieldPtr)
		return false;

    
	//Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=conFieldPtr->getDim();

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
				//con = (*conFieldPtr)[pt.x][pt.y][pt.z];
				con = conFieldPtr->get(pt);
				
			}
            //cerr<<"pt="<<pt<<" conc="<<con<<endl;
            Coordinates3D<double> coords(ptVec[0],ptVec[1],0); // notice that we are drawing pixels from other planes on a xy plan so we use ptVec instead of pt. pt is absolute position of the point ptVec is for projection purposes
			for (int idx=0 ; idx<4 ; ++idx){
			  Coordinates3D<double> cartesianVertex=cartesianVertices[idx]+coords; 
			  //cerr<<"cartesianVertex="<<cartesianVertex<<endl;
 			 _pointsArray->InsertNextPoint(cartesianVertex.x,cartesianVertex.y,0.0);
			 //cerr<<"after inserting into points array"<<endl;
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


std::vector<int> CAFieldExtractor::pointOrder(std::string _plane){
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
std::vector<int> CAFieldExtractor::dimOrder(std::string _plane){
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
