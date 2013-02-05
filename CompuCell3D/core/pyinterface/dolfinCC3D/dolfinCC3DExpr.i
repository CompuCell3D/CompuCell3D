

%module  dolfinCC3DExpr
//%module (directors="1") dolfin_compile_code_eccd61cf0689745bd6919adbb2c2b4c5

//%feature("director");

%{
#include <iostream>
 
#include <cmath>
#include <iostream>
#include <complex>
#include <stdexcept>
#include <numpy/arrayobject.h>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/Hierarchical.h>
#include <dolfin/common/types.h>
#include <dolfin/common/types.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/Variable.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/math/basic.h> 
 
#include <CustomExpressions.h> 
 
namespace dolfin
{
  
  // cmath functions
  using std::cos;
  using std::sin;
  using std::tan;
  using std::acos;
  using std::asin;
  using std::atan;
  using std::atan2;
  using std::cosh;
  using std::sinh;
  using std::tanh;
  using std::exp;
  using std::frexp;
  using std::ldexp;
  using std::log;
  using std::log10;
  using std::modf;
  using std::pow;
  using std::sqrt;
  using std::ceil;
  using std::fabs;
  using std::floor;
  using std::fmod;
  using std::max;
  using std::min;
  
  const double pi = DOLFIN_PI;
  
  class ExpressionNew: public Expression
  {
  public:
    double a;
    double b;
    ExpressionNew():Expression()
    {
  
      a = 0;
      b = 0;
    }
  
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const
    {
      values[0] = 1 + a*a + 2*b*b;
    }
  };
  
}
%}

//%feature("autodoc", "1");

%include <dolfin/common/Hierarchical.h>

%init%{

%}



%init%{
import_array();
%}

// Include global SWIG interface files:
// Typemaps, shared_ptr declarations, exceptions, version
%include <boost_shared_ptr.i>

%{
#define SWIG_SHARED_PTR_QNAMESPACE boost
%}



// Global typemaps
%include "dolfin/swig/typemaps/includes.i"

// Global exceptions
%include <exception.i>
%include "dolfin/swig/exceptions.i"

// Do not expand default arguments in C++ by generating two an extra 
// function in the SWIG layer. This reduces code bloat.
%feature("compactdefaultargs");

// STL SWIG string class
%include <std_string.i>

// Manually import ufc:
%shared_ptr(ufc::cell_integral)
%shared_ptr(ufc::dofmap)
%shared_ptr(ufc::finite_element)
%shared_ptr(ufc::function)
%shared_ptr(ufc::form)
%shared_ptr(ufc::exterior_facet_integral)
%shared_ptr(ufc::interior_facet_integral)
%import(module="ufc") "ufc.h"

// Local shared_ptr declarations
%shared_ptr(dolfin::Variable)
%shared_ptr(dolfin::GenericFunction)
%shared_ptr(dolfin::Expression)
%shared_ptr(dolfin::ExpressionNew)
%shared_ptr(dolfin::StepFunctionExpressionFlex)



// Import statements

// %import types from submodule common of SWIG module common
%include "dolfin/swig/common/pre.i"
%import(module="dolfin.cpp.common") "dolfin/common/types.h"
%import(module="dolfin.cpp.common") "dolfin/common/Array.h"
%import(module="dolfin.cpp.common") "dolfin/common/Variable.h"

// %import types from submodule function of SWIG module function
%include "dolfin/swig/function/pre.i"
%import(module="dolfin.cpp.function") "dolfin/function/GenericFunction.h"
%import(module="dolfin.cpp.function") "dolfin/function/Expression.h"
%import(module="dolfin.cpp.function") "dolfin/function/Function.h"
%import(module="dolfin.cpp.function") "dolfin/function/FunctionSpace.h"

%feature("autodoc", "1");


//
namespace dolfin
{
  
  // cmath functions
  using std::cos;
  using std::sin;
  using std::tan;
  using std::acos;
  using std::asin;
  using std::atan;
  using std::atan2;
  using std::cosh;
  using std::sinh;
  using std::tanh;
  using std::exp;
  using std::frexp;
  using std::ldexp;
  using std::log;
  using std::log10;
  using std::modf;
  using std::pow;
  using std::sqrt;
  using std::ceil;
  using std::fabs;
  using std::floor;
  using std::fmod;
  using std::max;
  using std::min;
  
  const double pi = DOLFIN_PI;
  
  class ExpressionNew: public Expression
  {
  public:
    double a;
    double b;
    ExpressionNew():Expression()
    {
  
      a = 0;
      b = 0;
    }
  
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const
    {
      values[0] = 1 + a*a + 2*b*b;
    }
  };
  
};

%include <CustomExpressions.h> 
