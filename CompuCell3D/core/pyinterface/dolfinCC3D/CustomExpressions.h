#ifndef CUSTOMEXPRESSIONS_H
#define CUSTOMEXPRESSIONS_H

#include <dolfin/common/Array.h>
// #include <dolfin/mesh/Mesh.h>
// #include <dolfin/mesh/SubDomain.h>
#include <CompuCell3D/Field3D/Dim3D.h>
#include <dolfin.h>
#include <vector>
#include <map>
#include <set>

namespace CompuCell3D{
    class Dim3D;    
    class Point3D;
    class CellG;
    template <class T> class Field3D;
    template <class T> class WatchableField3D;
    
}

namespace dolfin{
  
template <typename T> class Array;



class StepFunctionExpressionFlex: public Expression{
public:
  StepFunctionExpressionFlex(void * _cellField=0);
  ~StepFunctionExpressionFlex();
  void setCellField(void * _cellField);  
  
  void setStepFunctionValues(const std::map<unsigned char,double>& _cellTypeToValueMap=std::map<unsigned char,double>(), const std::map<long,double> & _cellIdsToValueMap=std::map<long,double>(),double _defaultExpressionValue=0.0);
  virtual void eval(Array<double>& values, const Array<double>& x) const;
  
private:
  CompuCell3D::WatchableField3D<CompuCell3D::CellG *> *cellField;
  CompuCell3D::Dim3D fieldDim;
  
    std::map<unsigned char,double> cellTypeToValueMap;
    std::map<long,double> cellIdsToValueMap;
    double defaultExpressionValue;
  
};




};

#endif