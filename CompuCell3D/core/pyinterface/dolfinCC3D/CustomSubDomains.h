#ifndef CUSTOMSUBDOMAINS_H
#define CUSTOMSUBDOMAINS_H

#include <dolfin/common/Array.h>
// #include <dolfin/mesh/Mesh.h>
// #include <dolfin/mesh/SubDomain.h>
#include <dolfin.h>
#include <vector>

namespace CompuCell3D{
    class Dim3D;
    class Point3D;
    class CellG;
    template <class T> class Field3D;
    template <class T> class WatchableField3D;
    
}

namespace dolfin{
  
template <typename T> class Array;

class OmegaCustom1: public SubDomain{
public:
  OmegaCustom1();
  ~OmegaCustom1();
  virtual bool inside(const Array<double>& x, bool on_boundary) const;    
//   using SubDomain::mark;
  
  
  
};

class OmegaCustom0: public SubDomain{
public:
  OmegaCustom0();
  ~OmegaCustom0();
  void setCellField(void * _cellField);
  void init();  
  virtual bool inside(const Array<double>& x, bool on_boundary) const;  
  
private:
  CompuCell3D::WatchableField3D<CompuCell3D::CellG *> *cellField;
  std::vector<CompuCell3D::Point3D> ptVec;;
  
};



};

#endif