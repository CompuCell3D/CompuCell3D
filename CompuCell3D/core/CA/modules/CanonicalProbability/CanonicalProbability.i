// -*-c++-*-


%module ("threads"=1) CanonicalProbability

%include "typemaps.i"

// ************************************************************
// Module Includes 
// ************************************************************

// These are copied directly to the .cxx file and are not parsed
// by SWIG.  Include include files or definitions that are required
// for the module to build correctly.
//DOCSTRINGS



%include <windows.i>

%{
// CompuCell3D Include Files
// #include <Potts3D/Cell.h>
#include <CA/ProbabilityFunction.h>
#include <CA/modules/CanonicalProbability/CanonicalProbability.h>
// #include <CompuCell3D/Field3D/Neighbor.h>
// #include <CompuCell3D/Boundary/BoundaryStrategy.h>
// #include <CompuCell3D/Field3D/Field3D.h>
// #include <CompuCell3D/Field3D/Field3DImpl.h>
// #include <CompuCell3D/Field3D/WatchableField3D.h>
// #include <CA/CACell.h>
// #include <CA/CAManager.h>
// #include <CA/CACellInventory.h>
// #include <CA/CACellStack.h>

// //necessary to get registration of change watcher working in Python
// #include <CompuCell3D/Field3D/Field3DChangeWatcher.h>
// #include <CA/CACellFieldChangeWatcher.h>


// //CA modules
// #include <CA/modules/CenterOfMassMonitor.h>



// // System Libraries
// #include <iostream>
// #include <sstream>
// #include <stdlib.h>


// #include "pyinterface/CompuCellPython/STLPyIterator.h"
// #include "pyinterface/CompuCellPython/STLPyIteratorMap.h"
// #include "pyinterface/CompuCellPython/STLPyIteratorRefRetType.h"


// Namespaces
using namespace std;
using namespace CompuCell3D;



%}




%include stl.i //to ensure stl functionality 

// // // %include "CompuCellExtraIncludes.i"

// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"

// C++ std::map handling
%include "std_set.i"

// C++ std::map handling
%include "std_vector.i"

%include "stl.i"

//enables better handling of STL exceptions
%include "exception.i"

%exception {
  try {
    $action
  } catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

// %include "swig_includes/numpy.i"
// // // %include "pyinterface/swig_includes/numpy.i"

// // // %init %{
    // // // import_array();
// // // %}


//C arrays
//%include "carrays.i"

// ******************************
// Third Party Classes
// ******************************
#define CANONICALPROBABILITY_EXPORT
%include <CA/ProbabilityFunction.h>
%include <CA/modules/CanonicalProbability/CanonicalProbability.h>

// // // %include "BasicUtils/BasicClassFactoryBase.h"
// // // %include "BasicUtils/BasicClassFactory.h"
// // // // %include <CompuCell3D/Plugin.h>
// // // // %include <BasicUtils/BasicPluginManager.h>

// // // // template <class T>
// // // // class BasicPluginManager {
// // // //   public:
// // // //     void loadLibraries(const std::string path);
// // // //     bool isLoaded(std::string pluginName);
// // // // };
// // // // 
// // // // %template(templatebasicpluginmanagersteppable) BasicPluginManager<Steppable>;
// // // // %template(templatebasicpluginmanagerplugin) BasicPluginManager<Plugin>;
// // // // 


// // // // ******************************
// // // // CompuCell3D Classes
// // // // ******************************


// // // //have to include all  export definitions for modules which are wrapped to avoid problems with interpreting by swig win32 specific c++ extensions...
// // // #define CASHARED_EXPORT
// // // #define BOUNDARYSHARED_EXPORT

// // // //modules #defines
// // // #define CENTEROFMASSMONITOR_EXPORT


// // // // %include <dolfin/mesh/Mesh.h>

// // // // we have to include files for objects that we will type-map before including definitions of corresponding typemaps
// // // %include "Field3D/Point3D.h"
// // // %include "Field3D/Dim3D.h"

// // // %extend CompuCell3D::Point3D{
  // // // std::string __str__(){
    // // // std::ostringstream s;
    // // // s<<(*self);
    // // // return s.str();
  // // // }
// // // };

// // // %extend CompuCell3D::Dim3D{
  // // // std::string __str__(){
    // // // std::ostringstream s;
    // // // s<<(*self);
    // // // return s.str();
  // // // }
// // // };

// // // // %extend CompuCell3D::Point3D{
  // // // // std::string __str__(){
    // // // // std::ostringstream s;
    // // // // s<<(*self);
    // // // // return s.str();
  // // // // }
  
// // // // %pythoncode %{
    // // // // def __getstate__(self):
        // // // // return (self.x,self.y,self.z)

    // // // // def __setstate__(self,tup):
        // // // // print 'tuple=',tup
        // // // // self.this = _CompuCell.new_Point3D(tup[0],tup[1],tup[2])
        // // // // self.thisown=1            
	
// // // // %}   
// // // // };


// // // // %extend CompuCell3D::Dim3D{
  // // // // std::string __str__(){
    // // // // std::ostringstream s;
    // // // // s<<(*self);
    // // // // return s.str();
  // // // // }
// // // // };

// // // %include <Utils/Coordinates3D.h>

// // // %template (Coordinates3DDouble) Coordinates3D<double>; 


// // // // %extend Coordinates3DDouble{
  // // // // std::string __str__(){
    // // // // std::ostringstream s;
    // // // // s<<(*self);
    // // // // return s.str();
  // // // // }
// // // // }  


// // // // %extend Coordinates3D<double>{
  // // // // std::string __str__(){
    // // // // std::ostringstream s;
    // // // // s<<"("<<(*self)<<")";
    // // // // return s.str();
  // // // // }
// // // // }  


// // // // turns on proper handling of default arguments - only one wrapper code will get generated for a function
// // // // alternative way could be to use typecheck maps but I had trouble with it.
// // // // compactdefaultargs has one disadvantage - it will not with all languages e.g Java and C# 
// // // // for more information see e.g. http://tech.groups.yahoo.com/group/swig/message/13432 
// // // %feature("compactdefaultargs"); 

// // // //typemaps for Point3D, Dim3D, Coordinates3D<double> - enable more convenient Python syntax e.g. Point3D can be specified as a list/tuple with 3 numerical elements
// // // // // // %include "typemaps_CC3D.i"

// // // %include <CompuCell3D/Field3D/Neighbor.h>
// // // %include <CompuCell3D/Boundary/BoundaryStrategy.h>


   
    
// // // using namespace CompuCell3D;


// // // %include "Field3D/Field3D.h"
// // // %include "Field3D/Field3DImpl.h"
// // // %include "Field3D/WatchableField3D.h"


// // // %include "CA/CACell.h"
// // // %include "CA/CACellStack.h"
// // // %include "CA/CACellInventory.h"
// // // %include "CA/CAManager.h"

// // // //Field3D<CACell*>
// // // %ignore Field3D<CACell*>::typeStr;
// // // %ignore Field3DImpl<CACell*>::typeStr;
// // // %ignore WatchableField3D<CACell*>::typeStr;

// // // %template(cellfield) Field3D<CACell *>;
// // // %template(cellfieldImpl) Field3DImpl<CACell *>;
// // // %template(watchablecellfield) WatchableField3D<CACell *>;

// // // //Field3D<CACellStack*>
// // // %ignore Field3D<CACellStack*>::typeStr;
// // // %ignore Field3DImpl<CACellStack*>::typeStr;
// // // %ignore WatchableField3D<CACellStack*>::typeStr;

// // // %template(cellstackfield) Field3D<CACellStack *>;
// // // %template(cellstackfieldImpl) Field3DImpl<CACellStack *>;
// // // %template(watchablecellstackfield) WatchableField3D<CACellStack *>;



// // // %include <CompuCell3D/Field3D/Field3DChangeWatcher.h>
// // // %template (Field3DChangeWatcherTemplate) Field3DChangeWatcher<CACell*>;

// // // %include <CA/CACellFieldChangeWatcher.h>


// // // %inline %{

// // // class STLPyIteratorCINV
// // // {
// // // public:

    // // // CompuCell3D::CACellInventory::cellInventoryContainerType::iterator current;
    // // // CompuCell3D::CACellInventory::cellInventoryContainerType::iterator begin;
    // // // CompuCell3D::CACellInventory::cellInventoryContainerType::iterator end;


    // // // STLPyIteratorCINV(CompuCell3D::CACellInventory::cellInventoryContainerType& a)
    // // // {
      // // // initialize(a);
    // // // }

    // // // STLPyIteratorCINV()
    // // // {
    // // // }

     // // // CompuCell3D::CACell * getCurrentRef(){
      // // // return const_cast<CompuCell3D::CACell * >(current->second);
      // // // // return const_cast<CompuCell3D::CellG * >(*current);
    // // // }
    // // // void initialize(CompuCell3D::CACellInventory::cellInventoryContainerType& a){
        // // // begin = a.begin();
        // // // end = a.end();
    // // // }
    // // // bool isEnd(){return current==end;}
    // // // bool isBegin(){return current==begin;}
    // // // void setToBegin(){current=begin;}

    // // // void previous(){
        // // // if(current != begin){

            // // // --current;
         // // // }

    // // // }

    // // // void next()
    // // // {

        // // // if(current != end){

            // // // ++current;
         // // // }


    // // // }
// // // };
// // // %}


// // // %template(vectorstring) std::vector<std::string>;
// // // %template(vectorCACellPtr) std::vector<CACell*>;


// // // //CA modules
// // // %include <CA/modules/CenterOfMassMonitor.h>

