
#include "CustomSubDomains.h"

#include <CompuCell3D/Field3D/Point3D.h>
#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <PublicUtilities/NumericalUtils.h>
#include <dolfin/common/Array.h>

using namespace dolfin;
using namespace std;

OmegaCustom1::OmegaCustom1(){}
OmegaCustom1::~OmegaCustom1(){}

bool OmegaCustom1::inside(const Array<double>& x, bool on_boundary) const
{
 return true; 
}

OmegaCustom0::OmegaCustom0():cellField(0){}
OmegaCustom0::~OmegaCustom0(){}

void OmegaCustom0::setCellField(void * _cellField){
 cellField=(CompuCell3D::WatchableField3D<CompuCell3D::CellG *> *)_cellField;
}

void OmegaCustom0::init(){
 using namespace CompuCell3D; 
 Point3D pt;
 CellG *cell;
 
 //can be optimizedfurther to visit only cell pixels instead of visiting all pixels
 Dim3D fieldDim=cellField->getDim();
 for (pt.x=0 ; pt.x<fieldDim.x ; ++pt.x)
   for (pt.y=0 ; pt.y<fieldDim.y ; ++pt.y)
     for (pt.z=0 ; pt.z<fieldDim.z ; ++pt.z){
	cell=cellField->get(pt);
	if(cell and cell->type==2){
	  ptVec.push_back(pt);
	}
     }
}



bool OmegaCustom0::inside(const Array<double>& x, bool on_boundary) const
{
 using namespace CompuCell3D;
 
  for (int i = 0 ; i < ptVec.size() ; ++i){
    Point3D pt=ptVec[i];
    if (x[0]>=pt.x && x[0]<=pt.x+1 && x[1]>=pt.y && x[1]<=pt.y+1 && x[2]>=pt.z && x[2]<=pt.z+1){
      return true;
    }
  }
  return false;  
  
}

SubdomainFlex::SubdomainFlex(void * _cellField):cellField(0){
  if(_cellField){
      cellField=reinterpret_cast<CompuCell3D::WatchableField3D<CompuCell3D::CellG *> *>(_cellField);
      fieldDim=cellField->getDim();
  }
}

SubdomainFlex::~SubdomainFlex(){}

void SubdomainFlex::setCellField(void * _cellField){
 cellField=reinterpret_cast<CompuCell3D::WatchableField3D<CompuCell3D::CellG *> *>(_cellField);
 fieldDim=cellField->getDim();
}

void SubdomainFlex::includeEntities(const std::vector<unsigned char>& _includeTypesVec, const std::vector<long> & _includeIdsVec){
  cellTypesSet.insert(_includeTypesVec.begin(),_includeTypesVec.end());
  cellIdsSet.insert(_includeIdsVec.begin(),_includeIdsVec.end());    
}


bool SubdomainFlex::inside(const Array<double>& x, bool on_boundary) const
{
  using namespace CompuCell3D;
  Point3D pt(short(round(x[0])),short(round(x[1])),short(round(x[2])));
  
  //keeping rounded point in the lattice
  if (pt.x==fieldDim.x)
      --pt.x;

  if (pt.y==fieldDim.y)
      --pt.y;

  if (pt.z==fieldDim.z)
      --pt.z;
  
  CellG *cell=cellField->get(pt);
  
  if(!cellTypesSet.size() && !cellIdsSet.size()) //when user does not specify any cell types or any cell ids tyhen we accume that entire mesh belongs to the inside
    return true;
    
  if (!cell){
     if (cellTypesSet.find(0)!=cellTypesSet.end())
       return true;     
  }else{
    if (cellTypesSet.find(cell->type)!=cellTypesSet.end())
      return true;
    else if (cellIdsSet.find(cell->id)!=cellIdsSet.end())
      return true;
  }
  return false;
}


