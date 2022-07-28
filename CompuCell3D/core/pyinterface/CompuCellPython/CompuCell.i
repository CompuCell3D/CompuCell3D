// -*-c++-*-


%module ("threads"=1, directors="1") CompuCell

// Have to replace ptrdiff_t with long long on windows. long on windows is 4 bytes
%apply long long {ptrdiff_t}




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
#define SWIG_FILE_WITH_INIT	

// CompuCell3D Include Files
// #include <Potts3D/Cell.h>
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
#include <CompuCell3D/Potts3D/CellInventoryWatcher.h>
#include <CompuCell3D/Potts3D/CellInventory.h>
#include <CompuCell3D/Potts3D/TypeChangeWatcher.h>
#include <CompuCell3D/Potts3D/TypeTransition.h>
#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
#include <CompuCell3D/Potts3D/EnergyFunctionCalculator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/steppables/PDESolvers/DiffusionSolverFE.h>
#include <CompuCell3D/steppables/PDESolvers/DiffusionSolverFE_CPU.h>
#include <CompuCell3D/steppables/PDESolvers/ReactionDiffusionSolverFE.h>

//#include <CompuCell3D/BabySim/BabyPottsParseData.h>
//#include <CompuCell3D/BabySim/BabySim.h>




//NeighborFinderParams
#include <NeighborFinderParams.h>

// Third Party Libraries
#include <PublicUtilities/NumericalUtils.h>
#include <PublicUtilities/Vector3.h>

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

//todo - numpy
#include <numpy/arrayobject.h>

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
////todo - numpy
%include "swig_includes/numpy.i"

%init %{
    import_array();
%}

// numpy array output from EnergyCalculator
%apply(bool* INPLACE_ARRAY1, int DIM1) { (bool * mask_array, size_t len) }
%apply(double* INPLACE_ARRAY1, int DIM1) { (double * double_array, size_t len) }
%apply(int* ARGOUT_ARRAY1, int DIM1) { (int* rangevec, int n) }
//%apply(bool* ARGOUT_ARRAY1, int DIM1) { (int* boolvec, int n) } // does not work
%apply(int* ARGOUT_ARRAY1, int DIM1) { (int* intvec, int n) }
%apply(double* ARGOUT_ARRAY1, int DIM1) { (double* doublevec, int n) }
%apply(short* ARGOUT_ARRAY1, int DIM1) { (short* shortvec, int n) }

//C arrays
//%include "carrays.i"


// ******************************
// CompuCell3D Classes
// ******************************


//have to include all  export definitions for modules which are arapped to avoid problems with interpreting by swig win32 specific c++ extensions...
#define COMPUCELLLIB_EXPORT

//#define BABYSIMLIB_EXPORT

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
//BiasVectorSteppable_autogenerated
#define BIASVECTORSTEPPABLE_EXPORT
//ImplicitMotility_autogenerated
#define IMPLICITMOTILITY_EXPORT
//CurvatureCalculator_autogenerated
#define CURVATURECALCULATOR_EXPORT
//OrientedGrowth_autogenerated
#define ORIENTEDGROWTH_EXPORT
//OrientedGrowth2_autogenerated
#define ORIENTEDGROWTH2_EXPORT
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

// we have to include files for objects that we will type-map before including definitions of corresponding typemaps
%include "Field3D/Point3D.h"
%include "Field3D/Dim3D.h"

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
        print( 'tuple=',tup)
        self.this = _CompuCell.new_Point3D(tup[0],tup[1],tup[2])
        self.thisown=1

    def to_tuple(self):
        return self.x, self.y, self.z

%}
};


%extend CompuCell3D::Dim3D{
  std::string __str__(){
    std::ostringstream s;
    s<<(*self);
    return s.str();
  }

%pythoncode %{
    def to_tuple(self):
        return self.x, self.y, self.z

%}

};

%include <Utils/Coordinates3D.h>

%template (Coordinates3DDouble) Coordinates3D<double>; 


// %extend Coordinates3DDouble{
  // std::string __str__(){
    // std::ostringstream s;
    // s<<(*self);
    // return s.str();
  // }
// }  


%extend Coordinates3D<double>{
  std::string __str__(){
    std::ostringstream s;
    s<<"("<<(*self)<<")";
    return s.str();
  }
}  


// turns on proper handling of default arguments - only one wrapper code will get generated for a function
// alternative way could be to use typecheck maps but I had trouble with it.
// compactdefaultargs has one disadvantage - it will not with all languages e.g Java and C# 
// for more information see e.g. http://tech.groups.yahoo.com/group/swig/message/13432 
%feature("compactdefaultargs"); 

//typemaps for Point3D, Dim3D, Coordinates3D<double> - enable more convenient Python syntax e.g. Point3D can be specified as a list/tuple with 3 numerical elements
%include "typemaps_CC3D.i"

%include <CompuCell3D/Field3D/Neighbor.h>
%include <CompuCell3D/Boundary/BoundaryStrategy.h>
%include "Potts3D/Cell.h"

    
    
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


    # simplifying access to cell Python dictionary
    def setdict(self,_dict):
        # raise AttributeError('ASSIGNMENT cell.dict=%s is illegal. dict can only be modified but not replaced'%(_dict))
        raise AttributeError('ASSIGNMENT cell.dict=%s is illegal. Dictionary "dict" can only be modified but not replaced'%(_dict))
        
    def getdict(self):
        dict_object = _CompuCell.getPyAttrib(self)
        return _CompuCell.getPyAttrib(self)
        
    __swig_setmethods__["dict"] = setdict
    __swig_getmethods__["dict"] = getdict

    if _newclass: dict = property(getdict,setdict)    
	
    # simplifying access to sbml models
    def setsbml(self, sbml) :		
        raise AttributeError('ASSIGNMENT cell.sbml = %s is illegal. '
                             '"sbml" attribute can only be modified but not replaced' % (sbml))

    def getsbml(self) :
        import weakref
        try:

            sbml_fetcher = self.dict['__sbml_fetcher']

            sbml_fetcher.cell_obj = weakref.ref(self)

            # sbml_fetcher.cell_ref = weakref.ref(self)

            return sbml_fetcher
        except (KeyError,AttributeError):

            class SBMLFetcher :
                def __init__(self, cell=None) :
                    import weakref
                    self.cell_id = -1
                    self.cell_obj = None
                    if cell is not None:
                        self.cell_id = cell.id



                def __getattr__(self, item) :
                    if item == 'cell_id':
                        return self.__dict__['cell_id']

                    cell_obj = self.cell_obj()
                    cell_dict = cell_obj.dict

                    try :
                        sbml_solver_dict = cell_dict['SBMLSolver']
                    except KeyError :
                        raise KeyError('Cell id={cell_id} has no SBML solvers'.format(cell_id = self.cell_id))

                    item_to_search = item
                    rr_flag = False
                    if item.startswith('_rr_'):
                        item_to_search = item[4:]
                        rr_flag = True

                    try :
                        rr_object =  sbml_solver_dict[item_to_search]
                    except KeyError :
                        raise KeyError('Could not find SBML solver with id={sbml_solver_id} in cell id={cell_id} '.format(
                            sbml_solver_id = item_to_search, cell_id = self.cell_id))

                    if rr_flag:
                        return rr_object
                    else:
                        return rr_object.model

            sbml_fetcher = SBMLFetcher(cell=self)
            self.dict['__sbml_fetcher'] = sbml_fetcher
            sbml_fetcher.cell_obj = weakref.ref(self)
            return sbml_fetcher


    __swig_setmethods__["sbml"] = setsbml
    __swig_getmethods__["sbml"] = getsbml

    if _newclass : sbml = property(getsbml, setsbml)

    __maboss__ = '__maboss__'
    
    def _get_maboss(self):
        cell_dict = self.dict
        class MaBoSSAccessor:
            def __getattr__(self, item):
                if CellG.__maboss__ not in cell_dict.keys():
                    cell_dict[CellG.__maboss__] = {}
                if item not in cell_dict[CellG.__maboss__].keys():
                    raise KeyError(f'Could not find MaBoSS solver with name {item}.')
                return cell_dict[CellG.__maboss__][item]
        return MaBoSSAccessor()

    def _set_maboss(self, val):
        raise AttributeError('ASSIGNMENT cell.maboss = %s is illegal. '
                             '"maboss" attribute can only be modified but not replaced' % (maboss))

    __swig_getmethods__["maboss"] = _get_maboss
    __swig_setmethods__["maboss"] = _set_maboss

    if _newclass : maboss = property(_get_maboss, _set_maboss)

      %}
    };

    



%include "Field3D/Field3D.h"
%include "Field3D/Field3DImpl.h"
%include "Field3D/WatchableField3D.h"



// %template(cellfield) CompuCell3D::Field3D<CellG *>;
// %template(floatfield) CompuCell3D::Field3D<float>;
// %template(floatfieldImpl) CompuCell3D::Field3DImpl<float>;
// %template(watchablecellfield) CompuCell3D::WatchableField3D<CellG *>;



%include <NeighborFinderParams.h>


%include <CompuCell3D/PluginManager.h>


// %include <CompuCell3D/Steppable.h>
// %include <CompuCell3D/Plugin.h>

%template(pluginmanagertemplate) CompuCell3D::PluginManager<Plugin> ;
%template(steppablemanagertemplate) CompuCell3D::PluginManager<Steppable> ;


// macros used to generate extra functions to better manipulate fields    
    
//#x
//Converts macro argument x to a string surrounded by double quotes ("x").
//x ## y
//Concatenates x and y together to form xy.
//`x`
//If x is a string surrounded by double quotes, do nothing. Otherwise, turn into a string like #x. This is a non-standard SWIG extension.    

double round(double d)
{
  return floor(d + 0.5);
}
    
//todo - numpy uncomment 
%define FIELD3DEXTENDERBASE(type,returnType)    
%extend  type{
    
  std::string __str__(){
    std::ostringstream s;
    s<<#type<<" dim"<<self->getDim();
    return s.str();
  }
  
  returnType min(){
    returnType minVal=self->get(Point3D(0,0,0));
    
    Dim3D dim=self->getDim();
    
    for (int x=0 ; x<dim.x ; ++x)
        for (int y=0 ; y<dim.y ; ++y)
            for (int z=0 ; z<dim.z ; ++z){
                returnType val = self->get(Point3D(x,y,z));
                if (val<minVal) minVal=val;            
            }
            
    return minVal;        
    
  }
  
  returnType max(){
    returnType maxVal=self->get(Point3D(0,0,0));
    
    Dim3D dim=self->getDim();
    
    for (int x=0 ; x<dim.x ; ++x)
        for (int y=0 ; y<dim.y ; ++y)
            for (int z=0 ; z<dim.z ; ++z){
                returnType val = self->get(Point3D(x,y,z));
                if (val>maxVal) maxVal=val;            
            }
            
    return maxVal;        
    
  }  
  
  returnType __getitem__(PyObject *_indexTuple) {
    if (!PyTuple_Check( _indexTuple) || PyTuple_GET_SIZE(_indexTuple)!=3){
        throw std::runtime_error(std::string(#type)+std::string(": Wrong Syntax: Expected someting like: field[1,2,3]"));
    }
    PyObject *xCoord=PyTuple_GetItem(_indexTuple,0);
    PyObject *yCoord=PyTuple_GetItem(_indexTuple,1);
    PyObject *zCoord=PyTuple_GetItem(_indexTuple,2);
    Py_ssize_t  x,y,z;

    //x-coord
    if (PyInt_Check(xCoord)){
        x=PyInt_AsLong(PyTuple_GetItem(_indexTuple,0));
    }else if (PyLong_Check(xCoord)){
        x=PyLong_AsLong(PyTuple_GetItem(_indexTuple,0));
    }else if (PyFloat_Check(xCoord)){
        x=(Py_ssize_t) round(PyFloat_AsDouble(PyTuple_GetItem(_indexTuple,0)));     
    }
    else{
        throw std::runtime_error("Wrong Type (X): only integer or float values are allowed here - floats are rounded");
    }    
    //y-coord
    if (PyInt_Check(yCoord)){
        y=PyInt_AsLong(PyTuple_GetItem(_indexTuple,1));
    }else if (PyLong_Check(yCoord)){
        y=PyLong_AsLong(PyTuple_GetItem(_indexTuple,1));
    }else if (PyFloat_Check(yCoord)){
        y = (Py_ssize_t) round(PyFloat_AsDouble(PyTuple_GetItem(_indexTuple,1)));     
    }
    else{
        throw std::runtime_error("Wrong Type (Y): only integer or float values are allowed here - floats are rounded");
    }    
    //z-coord    
    if (PyInt_Check(zCoord)){
        z=PyInt_AsLong(PyTuple_GetItem(_indexTuple,2));
    }else if (PyLong_Check(zCoord)){
        z=PyLong_AsLong(PyTuple_GetItem(_indexTuple,2));
    }else if (PyFloat_Check(zCoord)){
        z = (Py_ssize_t) round(PyFloat_AsDouble(PyTuple_GetItem(_indexTuple,2)));     
    }
    else{
        throw std::runtime_error("Wrong Type (Z): only integer or float values are allowed here - floats are rounded");
    }
    
    //cerr<<"x,y,z="<<x<<","<<y<<","<<z<<endl;
    return self->get(Point3D(x,y,z));    
  }

%enddef
    
%define FIELD3DEXTENDER(type,returnType)
FIELD3DEXTENDERBASE(type,returnType)    
  
%pythoncode %{

    def normalizeSlice(self, s):
        norm = lambda x : x if x is None else int(round(x))
        return slice ( norm(s.start),norm(s.stop), norm(s.step) )
        
    def __setitem__(self,_indexTyple,_val):
        newSliceTuple = tuple(map(lambda x : self.normalizeSlice(x) if isinstance(x,slice) else x , _indexTyple))  
        self.setitem(newSliceTuple,_val)

%}    
  
  void setitem(PyObject *_indexTuple,returnType _val) {
  // void __setitem__(PyObject *_indexTuple,returnType _val) {
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
		int ok = PySlice_GetIndices(xCoord, dim.x, &start_x, &stop_x, &step_x);

     // cout<<"extracting slices for x axis"<<endl;   
     //cerr<<"start x="<< start_x<<endl;
     //cerr<<"stop x="<< stop_x<<endl;
     //cerr<<"step x="<< step_x<<endl;
     //cerr<<"sliceLength="<<sliceLength<<endl;        
     //cerr<<"ok="<<ok<<endl;
        
    }else{        
        if (PyInt_Check(xCoord)){
            start_x=PyInt_AsLong(PyTuple_GetItem(_indexTuple,0));
            stop_x=start_x;
            step_x=1;
        }else if (PyLong_Check(xCoord)){
            start_x=PyLong_AsLong(PyTuple_GetItem(_indexTuple,0));
            stop_x=start_x;
            step_x=1;          
        }else if (PyFloat_Check(xCoord)){
            start_x = (Py_ssize_t) round(PyFloat_AsDouble(PyTuple_GetItem(_indexTuple,0)));     
            stop_x=start_x;
            step_x=1;                      
        }
        else{
            throw std::runtime_error("Wrong Type (X): only integer or float values are allowed here - floats are rounded");
        } 

        start_x %= dim.x;
        stop_x %= dim.x;
        stop_x += 1;

        if (start_x < 0)
            start_x = dim.x + start_x;

        if (stop_x < 0)
            stop_x = dim.x + stop_x;

    }

    if (PySlice_Check(yCoord)){
                
		int ok = PySlice_GetIndices(yCoord, dim.y, &start_y, &stop_y, &step_y);

        
    }else{
        if (PyInt_Check(yCoord)){
            start_y=PyInt_AsLong(PyTuple_GetItem(_indexTuple,1));
            stop_y=start_y;
            step_y=1;
        }else if (PyLong_Check(yCoord)){
            start_y=PyLong_AsLong(PyTuple_GetItem(_indexTuple,1));
            stop_y=start_y;
            step_y=1;          
        }else if (PyFloat_Check(yCoord)){
            start_y = (Py_ssize_t) round(PyFloat_AsDouble(PyTuple_GetItem(_indexTuple,1)));     
            stop_y=start_y;
            step_y=1;                      
        }
        else{
            throw std::runtime_error("Wrong Type (Y): only integer or float values are allowed here - floats are rounded");
        }

        start_y %= dim.y;
        stop_y %= dim.y;
        stop_y += 1;

        if (start_y < 0)
            start_y = dim.y + start_y;

        if (stop_y < 0)
            stop_y = dim.y + stop_y;

    }
    
    if (PySlice_Check(zCoord)){

	   int ok = PySlice_GetIndices(zCoord, dim.z, &start_z, &stop_z, &step_z);
        
    }else{
        if (PyInt_Check(zCoord)){
            start_z=PyInt_AsLong(PyTuple_GetItem(_indexTuple,2));
            stop_z=start_z;
            step_z=1;
        }else if (PyLong_Check(zCoord)){
            start_z=PyLong_AsLong(PyTuple_GetItem(_indexTuple,2));
            stop_z=start_z;
            step_z=1;          
        }else if (PyFloat_Check(zCoord)){
            start_z = (Py_ssize_t) round(PyFloat_AsDouble(PyTuple_GetItem(_indexTuple,2)));     
            stop_z=start_z;
            step_z=1;                      
        }
        else{
            throw std::runtime_error("Wrong Type (Z): only integer or float values are allowed here - floats are rounded");
        }
        start_z %= dim.z;
        stop_z %= dim.z;
        stop_z += 1;

        if (start_z < 0)
            start_z = dim.z + start_z;

        if (stop_z < 0)
            stop_z = dim.z + stop_z;


    }

   
    PyObject *sliceX=0,*sliceY=0,* sliceZ=0;
    
    //cout << "start_x, stop_x = " << start_x << "," << stop_x << endl;
    //cout << "start_y, stop_y = " << start_y << "," << stop_y << endl;
    //cout << "start_z, stop_z = " << start_z << "," << stop_z << endl;
    for (Py_ssize_t x=start_x ; x<stop_x ; x+=step_x)
        for (Py_ssize_t y=start_y ; y<stop_y ; y+=step_y)
            for (Py_ssize_t z=start_z ; z<stop_z ; z+=step_z){
                $self->set(Point3D(x,y,z),_val); 
            }
    
  }

  
}
%enddef    
    

//todo - numpy uncomment     
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

    def normalizeSlice(self, s):
        norm = lambda x : x if x is None else int(round(x))
        return slice ( norm(s.start),norm(s.stop), norm(s.step) )
        
    def __setitem__(self,_indexTyple,_val):
        newSliceTuple = tuple(map(lambda x : self.normalizeSlice(x) if isinstance(x,slice) else x , _indexTyple))  
        self.setitem(newSliceTuple,_val,self.volumeTrackerPlugin)

%}  

  void setitem(PyObject *_indexTuple,returnType _val,void *_volumeTrackerPlugin=0) {
    if (!PyTuple_Check( _indexTuple) || PyTuple_GET_SIZE(_indexTuple)!=3){
        throw std::runtime_error("Wrong Syntax: Expected someting like: field[1,2,3]=object");
    }
    
    //cerr << "setItem=" << endl;
    VolumeTrackerPlugin *volumeTrackerPlugin=(VolumeTrackerPlugin *)_volumeTrackerPlugin;
       
    
    PyObject *xCoord=PyTuple_GetItem(_indexTuple,0);
    PyObject *yCoord=PyTuple_GetItem(_indexTuple,1);
    PyObject *zCoord=PyTuple_GetItem(_indexTuple,2);
    
    Py_ssize_t  start_x, stop_x, step_x, sliceLength;
    Py_ssize_t  start_y, stop_y, step_y;
    Py_ssize_t  start_z, stop_z, step_z;
    
    Dim3D dim=self->getDim();
    
    if (PySlice_Check(xCoord)){    
        
		int ok = PySlice_GetIndices(xCoord, dim.x, &start_x, &stop_x, &step_x);
        
    }else{
        if (PyInt_Check(xCoord)){
            start_x=PyInt_AsLong(PyTuple_GetItem(_indexTuple,0));
            stop_x=start_x;
            step_x=1;
        }else if (PyLong_Check(xCoord)){
            start_x=PyLong_AsLong(PyTuple_GetItem(_indexTuple,0));
            stop_x=start_x;
            step_x=1;          
        }else if (PyFloat_Check(xCoord)){
            start_x = (Py_ssize_t) round(PyFloat_AsDouble(PyTuple_GetItem(_indexTuple,0)));     
            stop_x=start_x;
            step_x=1;                      
        }
        else{
            throw std::runtime_error("Wrong Type (X): only integer or float values are allowed here - floats are rounded");
        }   
        start_x %= dim.x;
        stop_x %= dim.x;
        stop_x += 1;

        if (start_x < 0)
            start_x = dim.x + start_x;

        if (stop_x < 0)
            stop_x = dim.x + stop_x;
    }

    if (PySlice_Check(yCoord)){       
		int ok = PySlice_GetIndices(yCoord, dim.y, &start_y, &stop_y, &step_y);

    }else{
        if (PyInt_Check(yCoord)){
            start_y=PyInt_AsLong(PyTuple_GetItem(_indexTuple,1));
            stop_y=start_y;
            step_y=1;
        }else if (PyLong_Check(yCoord)){
            start_y=PyLong_AsLong(PyTuple_GetItem(_indexTuple,1));
            stop_y=start_y;
            step_y=1;          
        }else if (PyFloat_Check(yCoord)){
            start_y = (Py_ssize_t) round(PyFloat_AsDouble(PyTuple_GetItem(_indexTuple,1)));     
            stop_y=start_y;
            step_y=1;                      
        }
        else{
            throw std::runtime_error("Wrong Type (Y): only integer or float values are allowed here - floats are rounded");
        }
        start_y %= dim.y;
        stop_y %= dim.y;
        stop_y += 1;

        if (start_y < 0)
            start_y = dim.y + start_y;

        if (stop_y < 0)
            stop_y = dim.y + stop_y;

    }
    
    if (PySlice_Check(zCoord)){                
		int ok = PySlice_GetIndices(zCoord, dim.z, &start_z, &stop_z, &step_z);        
        
    }else{
        if (PyInt_Check(zCoord)){
            start_z=PyInt_AsLong(PyTuple_GetItem(_indexTuple,2));
            stop_z=start_z;
            step_z=1;
        }else if (PyLong_Check(zCoord)){
            start_z=PyLong_AsLong(PyTuple_GetItem(_indexTuple,2));
            stop_z=start_z;
            step_z=1;          
        }else if (PyFloat_Check(zCoord)){
            start_z = (Py_ssize_t) round(PyFloat_AsDouble(PyTuple_GetItem(_indexTuple,2)));     
            stop_z=start_z;
            step_z=1;                      
        }
        else{
            throw std::runtime_error("Wrong Type (Z): only integer or float values are allowed here - floats are rounded");
        }
        start_z %= dim.z;
        stop_z %= dim.z;
        stop_z += 1;

        if (start_z < 0)
            start_z = dim.z + start_z;

        if (stop_z < 0)
            stop_z = dim.z + stop_z;

    }
   
    
//     cerr<<"start x="<< start_x<<endl;
//     cerr<<"stop x="<< stop_x<<endl;
//     cerr<<"step x="<< step_x<<endl;
//     cerr<<"sliceLength="<<sliceLength<<endl;
        
    //cout << "start_x, stop_x = " << start_x << "," << stop_x << endl;
    //cout << "start_y, stop_y = " << start_y << "," << stop_y << endl;
    //cout << "start_z, stop_z = " << start_z << "," << stop_z << endl;

    PyObject *sliceX=0,*sliceY=0,* sliceZ=0;
    
    for (Py_ssize_t x=start_x ; x<stop_x ; x+=step_x)
        for (Py_ssize_t y=start_y ; y<stop_y ; y+=step_y)
            for (Py_ssize_t z=start_z ; z<stop_z ; z+=step_z){
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

//todo - numpy uncomment 
CELLFIELD3DEXTENDER(Field3D<CellG *>,CellG*)
FIELD3DEXTENDER(Field3D<float>,float)
FIELD3DEXTENDER(Field3D<int>,int)



%template(vectorstdstring) std::vector<std::string>;
%template(vectordouble) std::vector<double>;
%template(vectorvectordouble) std::vector<std::vector<double> >;

%template(vectorint) std::vector<int>;
%template(vectorunsignedchar) std::vector<unsigned char>;
%template(vectorbool) std::vector<bool>;


%include "Field3D/Field3DChangeWatcher.h"
%template(cellgchangewatcher) CompuCell3D::Field3DChangeWatcher<CompuCell3D::CellG *>;



%template (mvectorCellGPtr) std::vector<CellG *>;
%template(mapLongCellGPtr)std::map<long,CellG *> ;
%template (mapLongFloat) std::map<long, float>;
%template(mapLongmapLongCellGPtr)std::map<long,std::map<long,CellG *> >;


%include "Potts3D/CellGChangeWatcher.h"
%include "Potts3D/TypeChangeWatcher.h"
%include "Potts3D/TypeTransition.h"
%include "Automaton/Automaton.h"
%include <CompuCell3D/Potts3D/CellInventoryWatcher.h>
%include <CompuCell3D/Potts3D/CellInventory.h>

%include <CompuCell3D/Potts3D/EnergyFunctionCalculator.h>
%include <CompuCell3D/Potts3D/Potts3D.h>

//%include <CompuCell3D/BabySim/BabyPottsParseData.h>
//%include <CompuCell3D/BabySim/BabySim.h>

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


//%include exception.i


// ************************************************************
// Init Functions
// ************************************************************


// ************************************************************
// Inline Functions
// ************************************************************

%{
struct CellInventoryWatcherForwarder : public CompuCell3D::CellInventoryWatcher {

    void *director;
    void (*directorFcnAdd)(CellG*, void*);
    void (*directorFcnRem)(CellG*, void*);

    CellInventoryWatcherForwarder(void (*_directorFcnAdd)(CellG*, void*), void (*_directorFcnRem)(CellG*, void*), void *_director) {
        director = _director;
        directorFcnAdd = _directorFcnAdd;
        directorFcnRem = _directorFcnRem;
    };
    ~CellInventoryWatcherForwarder() {
        if(director) {
            director = 0;
        }
    }
    void onCellAdd(CellG *cell) { directorFcnAdd(cell, director); }
    void onCellRemove(CellG *cell) { directorFcnRem(cell, director); }
};
%}

%feature("director") CellInventoryWatcherDir;

%inline %{

struct CellInventoryWatcherDir {
    
    CellInventoryWatcherForwarder *forwarder;

    CellInventoryWatcherDir() : forwarder{NULL} {};

    virtual void onCellAdd(CellG* cell) = 0;
    virtual void onCellRemove(CellG* cell) = 0;
    virtual ~CellInventoryWatcherDir() {
        if(forwarder) {
            delete forwarder;
            forwarder = 0;
        }
    }
};

%}

%{

struct CellInventoryWatcherEvaluator {
    CellInventoryWatcherDir *director;
};

void CellInventoryWatcherDirAdd(CellG *cell, void *_director) {
    CellInventoryWatcherDir *director = (CellInventoryWatcherDir*)_director;
    director->onCellAdd(cell);
}
void CellInventoryWatcherDirRem(CellG *cell, void *_director) {
    CellInventoryWatcherDir *director = (CellInventoryWatcherDir*)_director;
    director->onCellRemove(cell);
}

%}

%inline %{

void makeCellInventoryWatcher(CellInventoryWatcherDir *director, CellInventory *cInv) {
    director->forwarder = new CellInventoryWatcherForwarder(&CellInventoryWatcherDirAdd, &CellInventoryWatcherDirRem, director);
    cInv->registerWatcher(director->forwarder);
}

%}

%inline %{

   Field3D<float> * getConcentrationField(CompuCell3D::Simulator & simulator, std::string fieldName){
      std::map<std::string,Field3D<float>*> & fieldMap=simulator.getConcentrationFieldNameMap();
      std::map<std::string,Field3D<float>*>::iterator mitr;
      mitr=fieldMap.find(fieldName);
        
      if(mitr!=fieldMap.end()){
	Potts3D *potts = simulator.getPotts();  
	WatchableField3D<CellG *> *cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim(); 
    //set Dimensions of field    
	mitr->second->setDim(fieldDim); 
	return mitr->second;
      }else{
         return 0;
      }
   }
  
%}


%inline %{

    std::vector<std::string> getConcentrationFieldNames(CompuCell3D::Simulator & simulator) {
        std::map<std::string, Field3D<float>*> & fieldMap = simulator.getConcentrationFieldNameMap();
        std::map<std::string, Field3D<float>*>::iterator mitr;
        std::vector<std::string> field_names;
        for (mitr = fieldMap.begin(); mitr != fieldMap.end(); ++mitr)
            field_names.push_back(mitr->first);         

        return field_names;
    }

%}


%inline %{

void updateFluctuationCompensators() {

	if(Simulator::steppableManager.isLoaded("DiffusionSolverFE")) {
		DiffusionSolverFE_CPU * solver = (DiffusionSolverFE_CPU *)Simulator::steppableManager.get("DiffusionSolverFE");
		solver->updateFluctuationCompensator();
	}

	if(Simulator::steppableManager.isLoaded("ReactionDiffusionSolverFE")) {
		ReactionDiffusionSolverFE * solver = (ReactionDiffusionSolverFE *)Simulator::steppableManager.get("ReactionDiffusionSolverFE");
		solver->updateFluctuationCompensator();
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




  // todo - plugin manager
  void initializePlugins() {
    cerr << "initialize plugin fcn" << endl;

    char *steppablePath = getenv("COMPUCELL3D_STEPPABLE_PATH");
    cerr<<"steppablePath="<<steppablePath<<endl;
    if (steppablePath) Simulator::steppableManager.loadLibraries(steppablePath);
	  
    char *pluginPath = getenv("COMPUCELL3D_PLUGIN_PATH");
    cerr<<"pluginPath="<<pluginPath<<endl;
    cerr<<"THIS IS JUST BEFORE LOADING LIBRARIES"<<endl;
      
   
    if (pluginPath) Simulator::pluginManager.loadLibraries(pluginPath);

    //cerr<<" AFTER LOAD LIBRARIES"<<endl;
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
      cerr<<"ModuleName="<<source->moduleName<<endl;
   }

%}



%include "CompuCellExtraDeclarations.i"

%include "DerivedProperty.i"
