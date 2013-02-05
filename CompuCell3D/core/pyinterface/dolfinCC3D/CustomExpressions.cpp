
#include "CustomExpressions.h"

#include <CompuCell3D/Field3D/Point3D.h>
#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <PublicUtilities/NumericalUtils.h>

#include <dolfin/common/Array.h>
#include <iostream>

using namespace dolfin;
using namespace std;



StepFunctionExpressionFlex::StepFunctionExpressionFlex(void * _cellField):cellField(0),defaultExpressionValue(0.0){
  if(_cellField){
      cellField=reinterpret_cast<CompuCell3D::WatchableField3D<CompuCell3D::CellG *> *>(_cellField);
      fieldDim=cellField->getDim();
  }
}

StepFunctionExpressionFlex::~StepFunctionExpressionFlex(){}


void StepFunctionExpressionFlex::setCellField(void * _cellField){
  if(_cellField){
      cellField=reinterpret_cast<CompuCell3D::WatchableField3D<CompuCell3D::CellG *> *>(_cellField);
      fieldDim=cellField->getDim();
  }
//   cerr<<"cell field is "<<cellField<<std::endl;
  
}


void StepFunctionExpressionFlex::setStepFunctionValues(const std::map<unsigned char,double>& _cellTypeToValueMap, const std::map<long,double> & _cellIdsToValueMap,double _defaultExpressionValue){
    
    cellTypeToValueMap = _cellTypeToValueMap;
    cellIdsToValueMap = _cellIdsToValueMap;
    defaultExpressionValue = _defaultExpressionValue;
    
//     //debug output
//     cerr<<"cell type definitions "<<std::endl;
//     for(std::map<unsigned char,double>::const_iterator mitr=cellTypeToValueMap.begin() ; mitr != cellTypeToValueMap.end() ; ++mitr){
// 	cerr<<"type="<<mitr->first<<" value="<<mitr->second<<std::endl;
//     }
// 
//     
//     cerr<<"cell ids definitions "<<std::endl;
//     for(std::map<long,double>::const_iterator mitr=cellIdsToValueMap.begin() ; mitr != cellIdsToValueMap.end() ; ++mitr){
// 	cerr<<"type="<<mitr->first<<" value="<<mitr->second<<std::endl;
//     }
    
    
}


void StepFunctionExpressionFlex::eval(Array<double>& values, const Array<double>& x) const{
  using namespace CompuCell3D;
  Point3D pt(short(round(x[0])),short(round(x[1])),short(round(x[2])));

//   cerr<<"EVALUATING POINT "<<pt<<std::endl;
  //keeping rounded point in the lattice
  if (pt.x==fieldDim.x)
      --pt.x;

  if (pt.y==fieldDim.y)
      --pt.y;

  if (pt.z==fieldDim.z)
      --pt.z;
  
  CellG *cell=cellField->get(pt);
  
  if (!cellTypeToValueMap.size() && !cellIdsToValueMap.size()){
    values[0]= defaultExpressionValue;
    return;
 }

  //check Medium  
  if (!cell){
    std::map<unsigned char,double>::const_iterator ctMitr =cellTypeToValueMap.find(0); //have to use const_iterator becauee thr function is constant
    if (ctMitr!=cellTypeToValueMap.end()){
        values[0] = ctMitr->second; // found value for cell type 0
        return;
    }
  }else{
    //check if value for currenc CC3D cell type is defined
    std::map<unsigned char,double>::const_iterator ctMitr=cellTypeToValueMap.find(cell->type);
    if (ctMitr!=cellTypeToValueMap.end()){
        values[0] = ctMitr->second;
        return;
    }
    //if the value for this cell type is not defined, check if the value for its id is defined
    std::map<long,double>::const_iterator ciMitr=cellIdsToValueMap.find(cell->id);
    if (ciMitr!=cellIdsToValueMap.end()){
        values[0] = ciMitr->second;
        return;    
    }
    
  }  
  
  values[0]= defaultExpressionValue;
}


