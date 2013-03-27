// -*-c++-*-


%module ("threads"=1) CompuCell

%include "typemaps.i"

// ************************************************************
// Module Includes 
// ************************************************************

// These are copied directly to the .cxx file and are not parsed
// by SWIG.  Include include files or definitions that are required
// for the module to build correctly.
//DOCSTRINGS
%include "DocStrings.i"
   //Simulator.h
%feature("autodoc",Simulator_class) Simulator;
%feature("autodoc",getNumSteps_func) getNumSteps;
%feature("autodoc",getStep_func) getStep;
%feature("autodoc",isStepping_func) isStepping;
%feature("autodoc",getPotts_func) getPotts;

   //Potts.h
%feature("autodoc",Potts3D_class) Potts3D;
%feature("autodoc",getNumberOfAttempts_func) getNumberOfAttempts;
%feature("autodoc",getNumberOfAcceptedSpinFlips_func) getNumberOfAcceptedSpinFlip;
%feature("autodoc",getNumberOfAttemptedEnergyCalculations_func) getNumberOfAttemptedEnergyCalculations;
%feature("autodoc",getDepth_func) getDepth;
%feature("autodoc",setDepth_func) setDepth;
%feature("autodoc",setDebugOutputFrequency_func) setDebugOutputFrequency;
%feature("autodoc",getCellInventory_func) getCellInventory;

   //PottsParseData.h
%feature("autodoc",PottsParseData_Class) PottsParseData;

   //VolumeParseData.h
%feature("autodoc",VolumeParseData_Class) VolumeParseData; 
%feature("autodoc",TargetVolume_func) TargetVolume;
%feature("autodoc",LambdaVolume_func) LambdaVolume;




%include <windows.i>

%{
// CompuCell3D Include Files
#include <CompuCell3D/Plugin.h>
#include <CompuCell3D/Potts3D/Stepper.h>
#include <CompuCell3D/plugins/VolumeTracker/VolumeTrackerPlugin.h> //necesssary to make slicing in cellfield work
#include <CompuCell3D/Steppable.h>
#include <CompuCell3D/Field3D/Neighbor.h>
#include <CompuCell3D/Boundary/BoundaryStrategy.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/Field3DImpl.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <CompuCell3D/ClassRegistry.h>
#include <CompuCell3D/CC3DEvents.h>
#include <CompuCell3D/Simulator.h>

#include <CompuCell3D/PluginManager.h>
#include <CompuCell3D/Potts3D/CellInventory.h>
#include <CompuCell3D/Potts3D/TypeChangeWatcher.h>
#include <CompuCell3D/Potts3D/TypeTransition.h>
#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>

#include <CompuCell3D/Potts3D/Potts3D.h>
//NeighborFinderParams
#include <NeighborFinderParams.h>

// Third Party Libraries
#include <PublicUtilities/NumericalUtils.h>
#include <PublicUtilities/Vector3.h>

#include <BasicUtils/BasicException.h>
#include <BasicUtils/BasicSmartPointer.h>
#include <BasicUtils/BasicClassFactory.h>
#include <BasicUtils/BasicPluginManager.h>

// System Libraries
#include <iostream>
#include <stdlib.h>


#include "STLPyIterator.h"
#include "STLPyIteratorMap.h"
#include "STLPyIteratorRefRetType.h"


//EnergyFcns
//#include <CompuCell3D/Potts3D/EnergyFunction.h>
#include <EnergyFunctionPyWrapper.h>

#include <TypeChangeWatcherPyWrapper.h>

#include <CompuCell3D/Potts3D/AttributeAdder.h>


// Namespaces
using namespace std;
using namespace CompuCell3D;


%}


%include stl.i //to ensure stl functionality 

%include "CompuCellExtraIncludes.i"

// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"

// C++ std::map handling
%include "std_set.i"

// C++ std::map handling
%include "std_vector.i"


//enables better handling of STL exceptions
%include "exception.i"

%exception {
  try {
    $action
  } catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

//C arrays
//%include "carrays.i"

// ******************************
// Third Party Classes
// ******************************

%include "BasicUtils/BasicClassFactoryBase.h"
%include "BasicUtils/BasicClassFactory.h"
// %include <CompuCell3D/Plugin.h>
// %include <BasicUtils/BasicPluginManager.h>

// template <class T>
// class BasicPluginManager {
//   public:
//     void loadLibraries(const std::string path);
//     bool isLoaded(std::string pluginName);
// };
// 
// %template(templatebasicpluginmanagersteppable) BasicPluginManager<Steppable>;
// %template(templatebasicpluginmanagerplugin) BasicPluginManager<Plugin>;
// 


// ******************************
// CompuCell3D Classes
// ******************************


//have to include all  export definitions for modules which are arapped to avoid problems with interpreting by swig win32 specific c++ extensions...
#define COMPUCELLLIB_EXPORT
#define BOUNDARYSHARED_EXPORT
#define CHEMOTAXISSIMPLE_EXPORT
#define CHEMOTAXIS_EXPORT
#define MITOSIS_EXPORT
#define MITOSISSTEPPABLE_EXPORT
#define NEIGHBORTRACKER_EXPORT
#define PIXELTRACKER_EXPORT
#define BOUNDARYPIXELTRACKER_EXPORT
#define CONTACTLOCALFLEX_EXPORT
#define CONTACTLOCALPRODUCT_EXPORT
#define CONTACTMULTICAD_EXPORT
#define CELLORIENTATION_EXPORT
#define POLARIZATIONVECTOR_EXPORT
#define ELASTICITYTRACKER_EXPORT
#define ELASTICITY_EXPORT
#define PLASTICITYTRACKER_EXPORT
#define PLASTICITY_EXPORT

#define CONNECTIVITYLOCALFLEX_EXPORT
// #define LENGTHCONSTRAINTLOCALFLEX_EXPORT
#define LENGTHCONSTRAINT_EXPORT
#define MOLECULARCONTACT_EXPORT 
#define SECRETION_EXPORT 
#define VOLUMETRACKER_EXPORT 
#define CENTEROFMASS_EXPORT

//AutogeneratedModules - DO NOT REMOVE THIS LINE IT IS USED BY TWEDIT TO LOCATE CODE INSERTION POINT
//CleaverMeshDumper_autogenerated
#define CLEAVERMESHDUMPER_EXPORT
// // // //CGALMeshDumper_autogenerated
// // // #define CGALMESHDUMPER_EXPORT
//ContactOrientation_autogenerated
#define CONTACTORIENTATION_EXPORT
//BoundaryMonitor_autogenerated
#define BOUNDARYMONITOR_EXPORT
//CellTypeMonitor_autogenerated
#define CELLTYPEMONITOR_EXPORT
//Polarization23_autogenerated
#define POLARIZATION23_EXPORT
//ClusterSurface_autogenerated
#define CLUSTERSURFACE_EXPORT

//ClusterSurfaceTracker_autogenerated
#define CLUSTERSURFACETRACKER_EXPORT

// %include <dolfin/mesh/Mesh.h>


%include <CompuCell3D/Field3D/Neighbor.h>
%include <CompuCell3D/Boundary/BoundaryStrategy.h>
%include "Potts3D/Cell.h"




%include "Field3D/Point3D.h"
%include "Field3D/Dim3D.h"
%include "Field3D/Field3D.h"
%include "Field3D/Field3DImpl.h"
%include "Field3D/WatchableField3D.h"


%extend CompuCell3D::Point3D{
  std::string __str__(){
    std::ostringstream s;
    s<<(*self);
    return s.str();
  }
  
%pythoncode %{
    def __getstate__(self):
        return (self.x,self.y,self.z)

    def __setstate__(self,tup):
        print 'tuple=',tup
        self.this = _CompuCell.new_Point3D(tup[0],tup[1],tup[2])
        self.thisown=1            
	
%}   
};


%extend CompuCell3D::Dim3D{
  std::string __str__(){
    std::ostringstream s;
    s<<(*self);
    return s.str();
  }
};

// %template(cellfield) CompuCell3D::Field3D<CellG *>;
// %template(floatfield) CompuCell3D::Field3D<float>;
// %template(floatfieldImpl) CompuCell3D::Field3DImpl<float>;
// %template(watchablecellfield) CompuCell3D::WatchableField3D<CellG *>;



%include <NeighborFinderParams.h>


%include <CompuCell3D/PluginManager.h>

// template <class T>
// class PluginManager {
//   public:
//     void loadLibraries(const std::string path);
//     bool isLoaded(std::string pluginName);
// };

// %template(templatebasicpluginmanagersteppable) BasicPluginManager<Steppable>;
// %template(templatebasicpluginmanagerplugin) BasicPluginManager<Plugin>;


// %include <CompuCell3D/Steppable.h>
// %include <CompuCell3D/Plugin.h>
%include <BasicUtils/BasicPluginManager.h>

%template(bpmPlugin) BasicPluginManager<Plugin> ;
%template(bpmSteppable) BasicPluginManager<Steppable> ;

%template(pluginmanagertemplate) CompuCell3D::PluginManager<Plugin> ;
%template(steppablemanagertemplate) CompuCell3D::PluginManager<Steppable> ;


// %template(bpmStepNew) BasicPluginManager<StepNew> ;
// %template(stepnewmanagertemplate) CompuCell3D::PluginManager<StepNew> ;

%inline %{

  
  BasicPluginManager<Plugin> * getPluginManagerAsBPM(){
    return (BasicPluginManager<Plugin> *)&Simulator::pluginManager;
  }

  BasicPluginManager<Steppable> * getSteppableManagerAsBPM(){
    return (BasicPluginManager<Steppable> *)&Simulator::steppableManager;
  }
  
//   CompuCell3D::PluginManager<StepNew> getStepManager(){return CompuCell3D::PluginManager<StepNew>();}
//   CompuCell3D::PluginManager<Steppable> getSteppableManager(){return CompuCell3D::PluginManager<Steppable>();}
%}

using namespace CompuCell3D;



%extend CompuCell3D::CellG{
      %pythoncode %{
    def setVolume(self,_volume):
        raise AttributeError('ASSIGNMENT cell.volume=%s is illegal. volume is read only variable'%(_volume))

    __swig_setmethods__["volume"] = setVolume     
    if _newclass: volume = property(_CompuCell.CellG_volume_get,setVolume)
      
    def setSurface(self,_surface):
        raise AttributeError('ASSIGNMENT cell.surface=%s is illegal. surface is read only variable'%(_surface))

    __swig_setmethods__["surface"] = setSurface     
    if _newclass: surface = property(_CompuCell.CellG_surface_get,setSurface)

    def setxCM(self,_xCM):
        raise AttributeError('ASSIGNMENT cell.xCM=%s is illegal. xCM is read only variable'%(_xCM))
        
    __swig_setmethods__["xCM"] = setxCM     
    if _newclass: xCM = property(_CompuCell.CellG_xCM_get,setxCM)


    def setyCM(self,_yCM):
        raise AttributeError('ASSIGNMENT cell.yCM=%s is illegal. yCM is read only variable'%(_yCM))
        
    __swig_setmethods__["yCM"] = setyCM     
    if _newclass: yCM = property(_CompuCell.CellG_yCM_get,setyCM)
    
    def setzCM(self,_zCM):
        raise AttributeError('ASSIGNMENT cell.zCM=%s is illegal. zCM is read only variable'%(_zCM))
        
    __swig_setmethods__["zCM"] = setzCM     
    if _newclass: zCM = property(_CompuCell.CellG_zCM_get,setzCM)
    

    def setxCOM(self,_xCOM):
        raise AttributeError('ASSIGNMENT cell.xCOM=%s is illegal. xCOM is read only variable'%(_xCOM))
        
    __swig_setmethods__["xCOM"] = setxCOM     
    if _newclass: xCOM = property(_CompuCell.CellG_xCOM_get,setxCOM)
    
    def setyCOM(self,_yCOM):
        raise AttributeError('ASSIGNMENT cell.yCOM=%s is illegal. yCOM is read only variable'%(_yCOM))
        
    __swig_setmethods__["yCOM"] = setyCOM     
    if _newclass: yCOM = property(_CompuCell.CellG_yCOM_get,setyCOM)
    
    def setzCOM(self,_zCOM):
        raise AttributeError('ASSIGNMENT cell.zCOM=%s is illegal. zCOM is read only variable'%(_zCOM))
        
    __swig_setmethods__["zCOM"] = setzCOM     
    if _newclass: zCOM = property(_CompuCell.CellG_zCOM_get,setzCOM)
    
    def setxCOMPrev(self,_xCOMPrev):
        raise AttributeError('ASSIGNMENT cell.xCOMPrev=%s is illegal. xCOMPrev is read only variable'%(_xCOMPrev))
        
    __swig_setmethods__["xCOMPrev"] = setxCOMPrev     
    if _newclass: xCOMPrev = property(_CompuCell.CellG_xCOMPrev_get,setxCOMPrev)


    def setyCOMPrev(self,_yCOMPrev):
        raise AttributeError('ASSIGNMENT cell.yCOMPrev=%s is illegal. yCOMPrev is read only variable'%(_yCOMPrev))
        
    __swig_setmethods__["yCOMPrev"] = setyCOMPrev     
    if _newclass: yCOMPrev = property(_CompuCell.CellG_yCOMPrev_get,setyCOMPrev)
    
    def setzCOMPrev(self,_zCOMPrev):
        raise AttributeError('ASSIGNMENT cell.zCOMPrev=%s is illegal. zCOMPrev is read only variable'%(_zCOMPrev))
        
    __swig_setmethods__["zCOMPrev"] = setzCOMPrev     
    if _newclass: zCOMPrev = property(_CompuCell.CellG_zCOMPrev_get,setzCOMPrev)
    

    def setiXX(self,_iXX):
        raise AttributeError('ASSIGNMENT cell.iXX=%s is illegal. iXX is read only variable'%(_iXX))
        
    __swig_setmethods__["iXX"] = setiXX     
    if _newclass: iXX = property(_CompuCell.CellG_iXX_get,setiXX)
    

    def setiXY(self,_iXY):
        raise AttributeError('ASSIGNMENT cell.iXY=%s is illegal. iXY is read only variable'%(_iXY))
        
    __swig_setmethods__["iXY"] = setiXY     
    if _newclass: iXY = property(_CompuCell.CellG_iXY_get,setiXY)
    
    def setiXZ(self,_iXZ):
        raise AttributeError('ASSIGNMENT cell.iXZ=%s is illegal. iXZ is read only variable'%(_iXZ))
        
    __swig_setmethods__["iXZ"] = setiXZ     
    if _newclass: iXZ = property(_CompuCell.CellG_iXZ_get,setiXZ)
    
    
    def setiYY(self,_iYY):
        raise AttributeError('ASSIGNMENT cell.iYY=%s is illegal. iYY is read only variable'%(_iYY))
        
    __swig_setmethods__["iYY"] = setiYY     
    if _newclass: iYY = property(_CompuCell.CellG_iYY_get,setiYY)    
    
    def setiYZ(self,_iYZ):
        raise AttributeError('ASSIGNMENT cell.iYZ=%s is illegal. iYZ is read only variable'%(_iYZ))
        
    __swig_setmethods__["iYZ"] = setiYZ     
    if _newclass: iYZ = property(_CompuCell.CellG_iYZ_get,setiYZ)    
    
    def setiZZ(self,_iZZ):
        raise AttributeError('ASSIGNMENT cell.iZZ=%s is illegal. iZZ is read only variable'%(_iZZ))
        
    __swig_setmethods__["iZZ"] = setiZZ     
    if _newclass: iZZ = property(_CompuCell.CellG_iZZ_get,setiZZ)    
    
    def setlX(self,_lX):
        raise AttributeError('ASSIGNMENT cell.lX=%s is illegal. lX is read only variable'%(_lX))
        
    __swig_setmethods__["lX"] = setlX     
    if _newclass: lX = property(_CompuCell.CellG_lX_get,setlX)

    def setlY(self,_lY):
        raise AttributeError('ASSIGNMENT cell.lY=%s is illegal. lY is read only variable'%(_lY))
        
    __swig_setmethods__["lY"] = setlY     
    if _newclass: lY = property(_CompuCell.CellG_lY_get,setlY)
    
    def setlX(self,_lX):
        raise AttributeError('ASSIGNMENT cell.lX=%s is illegal. lX is read only variable'%(_lX))
        	
    __swig_setmethods__["lX"] = setlX     
    if _newclass: lX = property(_CompuCell.CellG_lX_get,setlX)
    
    def setecc(self,_ecc):
        raise AttributeError('ASSIGNMENT cell.ecc=%s is illegal. ecc is read only variable'%(_ecc))
        
    __swig_setmethods__["ecc"] = setecc     
    if _newclass: ecc = property(_CompuCell.CellG_ecc_get,setecc)
    
    def setid(self,_id):
        raise AttributeError('ASSIGNMENT cell.id=%s is illegal. id is read only variable'%(_id))
        
    __swig_setmethods__["id"] = setid     
    if _newclass: id = property(_CompuCell.CellG_id_get,setid)
    
    def setclusterId(self,_clusterId):
        raise AttributeError('ASSIGNMENT cell.clusterId=%s is illegal. Please use self.inventory.reassignClusterId(cell,NEW_CLUSTER_ID) function'%(_clusterId))
        
    __swig_setmethods__["clusterId"] = setclusterId     
    if _newclass: clusterId = property(_CompuCell.CellG_clusterId_get,setclusterId)
    
    
    def setextraAttribPtr(self,_extraAttribPtr):
        raise AttributeError('ASSIGNMENT cell.extraAttribPtr=%s is illegal. extraAttribPtr is read only variable'%(_extraAttribPtr))
        
    __swig_setmethods__["extraAttribPtr"] = setextraAttribPtr     
    if _newclass: extraAttribPtr = property(_CompuCell.CellG_extraAttribPtr_get,setextraAttribPtr)
    
    def setpyAttrib(self,_pyAttrib):
        raise AttributeError('ASSIGNMENT cell.pyAttrib=%s is illegal. pyAttrib is read only variable'%(_pyAttrib))
        
    __swig_setmethods__["pyAttrib"] = setpyAttrib     
    if _newclass: pyAttrib = property(_CompuCell.CellG_pyAttrib_get,setpyAttrib)
    
      %}
    };

// macros used to generate extra functions to better manipulate fields    
    
//#x
//Converts macro argument x to a string surrounded by double quotes ("x").
//x ## y
//Concatenates x and y together to form xy.
//`x`
//If x is a string surrounded by double quotes, do nothing. Otherwise, turn into a string like #x. This is a non-standard SWIG extension.    
    
%define FIELD3DEXTENDERBASE(type,returnType)    
%extend  type{
    
  std::string __str__(){
    std::ostringstream s;
    s<<#type<<" dim"<<self->getDim();
    return s.str();
  }
  
  returnType __getitem__(PyObject *_indexTuple) {
    if (!PyTuple_Check( _indexTuple) || PyTuple_GET_SIZE(_indexTuple)!=3){
        throw std::runtime_error(std::string(#type)+std::string(": Wrong Syntax: Expected someting like: field[1,2,3]"));
    }

    return self->get(Point3D(PyInt_AsLong(PyTuple_GetItem(_indexTuple,0)),PyInt_AsLong(PyTuple_GetItem(_indexTuple,1)),PyInt_AsLong(PyTuple_GetItem(_indexTuple,2))));    
  }

%enddef
    
%define FIELD3DEXTENDER(type,returnType)
FIELD3DEXTENDERBASE(type,returnType)    
  
  void __setitem__(PyObject *_indexTuple,returnType _val) {
    if (!PyTuple_Check( _indexTuple) || PyTuple_GET_SIZE(_indexTuple)!=3){
        throw std::runtime_error("Wrong Syntax: Expected someting like: field[1,2,3]=object");
    }
    
    PyObject *xCoord=PyTuple_GetItem(_indexTuple,0);
    PyObject *yCoord=PyTuple_GetItem(_indexTuple,1);
    PyObject *zCoord=PyTuple_GetItem(_indexTuple,2);
    
    Py_ssize_t  start_x, stop_x, step_x, sliceLength;
    Py_ssize_t  start_y, stop_y, step_y;
    Py_ssize_t  start_z, stop_z, step_z;
    
    Dim3D dim=self->getDim();
    
    if (PySlice_Check(xCoord)){

        PySlice_GetIndicesEx((PySliceObject*)xCoord,dim.x-1,&start_x,&stop_x,&step_x,&sliceLength);

        
    }else{
        start_x=PyInt_AsLong(PyTuple_GetItem(_indexTuple,0));
        stop_x=start_x;
        step_x=1;
    }

    if (PySlice_Check(yCoord)){
        
        PySlice_GetIndicesEx((PySliceObject*)yCoord,dim.y-1,&start_y,&stop_y,&step_y,&sliceLength);
        
        
    }else{
        start_y=PyInt_AsLong(PyTuple_GetItem(_indexTuple,1));
        stop_y=start_y;
        step_y=1;
    }
    
    if (PySlice_Check(zCoord)){
        
        PySlice_GetIndicesEx((PySliceObject*)zCoord,dim.z-1,&start_z,&stop_z,&step_z,&sliceLength);
        
        
    }else{
        start_z=PyInt_AsLong(PyTuple_GetItem(_indexTuple,2));
        stop_z=start_z;
        step_z=1;
    }

    
//     cerr<<"start x="<< start_x<<endl;
//     cerr<<"stop x="<< stop_x<<endl;
//     cerr<<"step x="<< step_x<<endl;
//     cerr<<"sliceLength="<<sliceLength<<endl;
    
    
    int x,y,z;
    PyObject *sliceX=0,*sliceY=0,* sliceZ=0;
    
    for (Py_ssize_t x=start_x ; x<=stop_x ; x+=step_x)
        for (Py_ssize_t y=start_y ; y<=stop_y ; y+=step_y)
            for (Py_ssize_t z=start_z ; z<=stop_z ; z+=step_z){
                $self->set(Point3D(x,y,z),_val); 
            }
    
  }

  
}
%enddef    
    

    
%define CELLFIELD3DEXTENDER(type,returnType)
FIELD3DEXTENDERBASE(type,returnType)    
// %extend  type{
//     
//   std::string __str__(){
//     std::ostringstream s;
//     s<<#type<<" dim"<<self->getDim();
//     return s.str();
//   }
//   
//   returnType __getitem__(PyObject *_indexTuple) {
//     if (!PyTuple_Check( _indexTuple) || PyTuple_GET_SIZE(_indexTuple)!=3){
//         throw std::runtime_error(std::string(#type)+std::string(": Wrong Syntax: Expected someting like: field[1,2,3]"));
//     }
// 
//     return self->get(Point3D(PyInt_AsLong(PyTuple_GetItem(_indexTuple,0)),PyInt_AsLong(PyTuple_GetItem(_indexTuple,1)),PyInt_AsLong(PyTuple_GetItem(_indexTuple,2))));    
//   }
  
//for cell field we call __setitem__ Python implementation which in turn calls setitem c++ implementation. We do this to pass volumeTracker plugin
//to setitem (c++) so that after each set operation on the field we can check if ther is any cell which needs to be deleted (this is what volumeTrackerPlugin step fcn does)
// otherwise we might end up with memory leaks.
//We could have also included volumeTrackerPlugin ptr in WatchableField3D<> but decided that it woudl be better not to ploute main code - might reconsider this though

%pythoncode %{
    def __setitem__(self,_indexTyple,_val):
        self.setitem(_indexTyple,_val,self.volumeTrackerPlugin)
%}  

  void setitem(PyObject *_indexTuple,returnType _val,void *_volumeTrackerPlugin=0) {
    if (!PyTuple_Check( _indexTuple) || PyTuple_GET_SIZE(_indexTuple)!=3){
        throw std::runtime_error("Wrong Syntax: Expected someting like: field[1,2,3]=object");
    }
    
    VolumeTrackerPlugin *volumeTrackerPlugin=(VolumeTrackerPlugin *)_volumeTrackerPlugin;
    
    PyObject *xCoord=PyTuple_GetItem(_indexTuple,0);
    PyObject *yCoord=PyTuple_GetItem(_indexTuple,1);
    PyObject *zCoord=PyTuple_GetItem(_indexTuple,2);
    
    Py_ssize_t  start_x, stop_x, step_x, sliceLength;
    Py_ssize_t  start_y, stop_y, step_y;
    Py_ssize_t  start_z, stop_z, step_z;
    
    Dim3D dim=self->getDim();
    
    if (PySlice_Check(xCoord)){

        PySlice_GetIndicesEx((PySliceObject*)xCoord,dim.x-1,&start_x,&stop_x,&step_x,&sliceLength);

        
    }else{
        start_x=PyInt_AsLong(PyTuple_GetItem(_indexTuple,0));
        stop_x=start_x;
        step_x=1;
    }

    if (PySlice_Check(yCoord)){
        
        PySlice_GetIndicesEx((PySliceObject*)yCoord,dim.y-1,&start_y,&stop_y,&step_y,&sliceLength);
        
        
    }else{
        start_y=PyInt_AsLong(PyTuple_GetItem(_indexTuple,1));
        stop_y=start_y;
        step_y=1;
    }
    
    if (PySlice_Check(zCoord)){
        
        PySlice_GetIndicesEx((PySliceObject*)zCoord,dim.z-1,&start_z,&stop_z,&step_z,&sliceLength);
        
        
    }else{
        start_z=PyInt_AsLong(PyTuple_GetItem(_indexTuple,2));
        stop_z=start_z;
        step_z=1;
    }

    
//     cerr<<"start x="<< start_x<<endl;
//     cerr<<"stop x="<< stop_x<<endl;
//     cerr<<"step x="<< step_x<<endl;
//     cerr<<"sliceLength="<<sliceLength<<endl;
    
    
    int x,y,z;
    PyObject *sliceX=0,*sliceY=0,* sliceZ=0;
    
    for (Py_ssize_t x=start_x ; x<=stop_x ; x+=step_x)
        for (Py_ssize_t y=start_y ; y<=stop_y ; y+=step_y)
            for (Py_ssize_t z=start_z ; z<=stop_z ; z+=step_z){
                $self->set(Point3D(x,y,z),_val);
                volumeTrackerPlugin->step();
            }
    

  }

  
}
%enddef    
    
    
    
%ignore Field3D<float>::typeStr;
%ignore Field3DImpl<float>::typeStr;
%ignore Field3D<int>::typeStr;
%ignore Field3DImpl<int>::typeStr;
%ignore Field3D<CellG*>::typeStr;
%ignore Field3DImpl<CellG*>::typeStr;
%ignore WatchableField3D<CellG*>::typeStr;

%template(floatfield) Field3D<float>;
%template(floatfieldImpl) Field3DImpl<float>;
%template(intfield) Field3D<int>;
%template(intfieldImpl) Field3DImpl<int>;

%template(cellfield) Field3D<CellG *>;
%template(cellfieldImpl) Field3DImpl<CellG *>;
%template(watchablecellfield) WatchableField3D<CellG *>;


CELLFIELD3DEXTENDER(Field3D<CellG *>,CellG*)
FIELD3DEXTENDER(Field3D<float>,float)
FIELD3DEXTENDER(Field3D<int>,int)



%template(vectorstdstring) std::vector<std::string>;
%template(vectordouble) std::vector<double>;

%template(vectorint) std::vector<int>;



%include "Field3D/Field3DChangeWatcher.h"
%template(cellgchangewatcher) CompuCell3D::Field3DChangeWatcher<CompuCell3D::CellG *>;

%template(vectorCellGPtr)std::vector<CompuCell3D::CellG *> ;
%template(mapLongCellGPtr)std::map<long,CellG *> ;
%template(mapLongmapLongCellGPtr)std::map<long,std::map<long,CellG *> >;


%include "Potts3D/CellGChangeWatcher.h"
%include "Potts3D/TypeChangeWatcher.h"
%include "Potts3D/TypeTransition.h"
%include "Automaton/Automaton.h"
%include <CompuCell3D/Potts3D/CellInventory.h>


%include <CompuCell3D/Potts3D/Potts3D.h>

%include "Steppable.h"
%include "ClassRegistry.h"
%include <CompuCell3D/SteerableObject.h>
%include "Simulator.h"
%include <CompuCell3D/CC3DEvents.h>


%include <CompuCell3D/ParseData.h>
%include <CompuCell3D/ParserStorage.h>
%include <CompuCell3D/PottsParseData.h>

%include <PublicUtilities/NumericalUtils.h>
%include <PublicUtilities/Vector3.h>

%include <BasicUtils/BasicException.h>


//%include exception.i


// ************************************************************
// Init Functions
// ************************************************************


// ************************************************************
// Inline Functions
// ************************************************************


%inline %{

   Field3D<float> * getConcentrationField(CompuCell3D::Simulator & simulator, std::string fieldName){
      std::map<std::string,Field3D<float>*> & fieldMap=simulator.getConcentrationFieldNameMap();
      std::map<std::string,Field3D<float>*>::iterator mitr;
      mitr=fieldMap.find(fieldName);
        
      if(mitr!=fieldMap.end()){
	Potts3D *potts = simulator.getPotts();  
	WatchableField3D<CellG *> *cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim(); 
	mitr->second->setDim(fieldDim); //set Dimensions of field
	return mitr->second;
      }else{
         return 0;
      }
   }
  

%}

%inline %{

   bool areCellsDifferent(CellG *_cell1,CellG *_cell2){
      return _cell1 != _cell2;
   }
   
   CellG* getMediumCell(){
    return (CellG*)0;
   }
   
%}

%include "STLPyIterator.h"
%include "STLPyIteratorMap.h"

%include "STLPyIteratorRefRetType.h"

%template (cellInvPyItr) STLPyIterator<CompuCell3D::CellInventory::cellInventoryContainerType>;
//iterators for maps - we cannot rely on swig generated iterators for STL containers when we have multiple modules
%template(mapLongCellGPtr)std::map<long,CellG *> ;
// %template (mapLongCellGPtrPyItr) STLPyIteratorRefRetType<std::map<long, CompuCell3D::CellG *> , std::pair<long,CellG *> >;
%template (pairLongCellGPtr) std::pair<long const, CompuCell3D::CellG *>;
%template (mapLongCellGPtrPyItr) STLPyIteratorMap<std::map<long, CompuCell3D::CellG *>, CompuCell3D::CellG *>;
%template (compartmentinventoryPtrPyItr) STLPyIteratorMap<CompuCell3D::CompartmentInventory::compartmentInventoryContainerType, CompuCell3D::CompartmentInventory::compartmentListContainerType>;
// %template (mapLongCellGPtrPyItr) STLPyIterator<std::map<long, CompuCell3D::CellG *> >;


%inline %{

class STLPyIteratorCINV
{
public:

    CompuCell3D::CellInventory::cellInventoryContainerType::iterator current;
    CompuCell3D::CellInventory::cellInventoryContainerType::iterator begin;
    CompuCell3D::CellInventory::cellInventoryContainerType::iterator end;


    STLPyIteratorCINV(CompuCell3D::CellInventory::cellInventoryContainerType& a)
    {
      initialize(a);
    }

    STLPyIteratorCINV()
    {
    }


     CompuCell3D::CellG * getCurrentRef(){
      return const_cast<CompuCell3D::CellG * >(current->second);
      // return const_cast<CompuCell3D::CellG * >(*current);
    }
    void initialize(CompuCell3D::CellInventory::cellInventoryContainerType& a){
        begin = a.begin();
        end = a.end();
    }
    bool isEnd(){return current==end;}
    bool isBegin(){return current==begin;}
    void setToBegin(){current=begin;}

    void previous(){
        if(current != begin){

            --current;
         }

    }

    void next()
    {

        if(current != end){

            ++current;
         }


    }
};

//STL iterator for compartments of a cluster 

class STLPyIteratorCOMPARTMENT
{
public:

    CompuCell3D::CompartmentInventory::compartmentInventoryContainerType::iterator current;
    CompuCell3D::CompartmentInventory::compartmentInventoryContainerType::iterator begin;
    CompuCell3D::CompartmentInventory::compartmentInventoryContainerType::iterator end;


    STLPyIteratorCOMPARTMENT(CompuCell3D::CompartmentInventory::compartmentInventoryContainerType& a)
    {
      initialize(a);
    }

    STLPyIteratorCOMPARTMENT()
    {
    }


//     CompuCell3D::CompartmentInventory::compartmentListContainerType * getCurrentRef(){
  //    return &(const_cast<CompuCell3D::CompartmentInventory::compartmentListContainerType>(current->second));      
    //}
    
     CompuCell3D::CompartmentInventory::compartmentListContainerType & getCurrentRef(){
		return current->second;
		//CompuCell3D::CompartmentInventory::compartmentListContainerType & compartmentListRef=current->second;
		//return compartmentListRef;
        //return const_cast<std::map<long,CellG *> >(current->second);      
    }
    
     
    
    void initialize(CompuCell3D::CompartmentInventory::compartmentInventoryContainerType& a){
        begin = a.begin();
        end = a.end();
    }
    bool isEnd(){return current==end;}
    bool isBegin(){return current==begin;}
    void setToBegin(){current=begin;}

    void previous(){
        if(current != begin){

            --current;
         }

    }

    void next()
    {

        if(current != end){

            ++current;
         }


    }
};



%}


%inline %{

  /**
   * Wrap the plugin initialization code from main.cpp.
   * This must be called before the config file is loaded and any
   * plugins are used.
   */

  void initializePlugins() {
    // Set the path and load the plugins

    char *steppablePath = getenv("COMPUCELL3D_STEPPABLE_PATH");
    cerr<<"steppablePath="<<steppablePath<<endl;
    if (steppablePath) Simulator::steppableManager.loadLibraries(steppablePath);

    char *pluginPath = getenv("COMPUCELL3D_PLUGIN_PATH");
    cerr<<"pluginPath="<<pluginPath<<endl;
    cerr<<"THIS IS JUST BEFORE LOADING LIBRARIES"<<endl;
      
   
    if (pluginPath) Simulator::pluginManager.loadLibraries(pluginPath);

    cerr<<" AFTER LOAD LIBRARIES"<<endl;
    // Display the plugins that were loaded

  }



  /**
   * Read and parse the config file.  This should be called after
   * the plugins have been initialized.
   *
   * @param sConfig the path to the config file
   */



  void assignParseDataPtr(ParseData * source, ParseData * destination){
      destination=source;
   }

   void printModuleName(ParseData * source){
      cerr<<"moduleName="<<source->moduleName<<endl;
   }

%}

%include "CompuCellExtraDeclarations.i"
