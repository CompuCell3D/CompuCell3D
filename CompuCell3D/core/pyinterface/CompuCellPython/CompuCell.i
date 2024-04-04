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
#include <CompuCell3D/CC3DExceptions.h>

#include <CompuCell3D/PluginManager.h>
#include <CompuCell3D/Potts3D/CellInventoryWatcher.h>
#include <CompuCell3D/Potts3D/CellInventory.h>
#include <CompuCell3D/Potts3D/TypeChangeWatcher.h>
#include <CompuCell3D/Potts3D/TypeTransition.h>
#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
#include <CompuCell3D/Potts3D/EnergyFunctionCalculator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/steppables/PDESolvers/DiffusionSolverFE.h>
#include<CompuCell3D/steppables/PDESolvers/ReactionDiffusionSolverFVM.h>
#include <CompuCell3D/steppables/PDESolvers/DiffusionSolverFE_CPU.h>
#include <CompuCell3D/steppables/PDESolvers/ReactionDiffusionSolverFE.h>
#include <Logger/CC3DLogger.h>
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

%include "CompuCell3D/CC3DExceptions.h"

%exception {
  try {
    $action
  } catch (const CompuCell3D::CC3DException& e){
      std::string msg = "C++ CC3DException in " + e.getFilename() + ": " + e.getMessage();
      PyErr_SetString(PyExc_RuntimeError, msg.c_str());
      SWIG_fail;
      return nullptr;

//      cerr<<"CAUGHT CC3DEXCEPTION"<<msg<<endl;
//      SWIG_exception(SWIG_RuntimeError, msg.c_str());
  } catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
    SWIG_fail;
    return nullptr;
    }
//  catch (...){
//      SWIG_exception(SWIG_RuntimeError, "Unknown exception in C++ code");
//  }
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
#define LOGGER_EXPORT
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

// PDE Solvers
#define PDESOLVERS_EXPORT

// Logger
#define CAPI_EXPORT
// %include <dolfin/mesh/Mesh.h>

// we have to include files for objects that we will type-map before including definitions of corresponding typemaps
// logger include
%include "Logger/CC3DLogger.h"
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

    def __reduce__(self):
        return Dim3D, (self.x, self.y, self.z)
%}
};

%include <Utils/Coordinates3D.h>

%template (Coordinates3DDouble) Coordinates3D<double>; 



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

//typemaps for Point3D, Dim3D, Coordinates3D<double> - enable more convenient Python syntax e.g.
// Point3D can be specified as a list/tuple with 3 numerical elements
%include "typemaps_CC3D.i"

%include <CompuCell3D/Field3D/Neighbor.h>
%include <CompuCell3D/Boundary/BoundaryStrategy.h>
%include "Potts3D/Cell.h"


    
    
using namespace CompuCell3D;

%define CELLG_ATTRIB(attribName)
    %pythoncode %{
        def set_ ## attribName(self, ## attribName):
            raise AttributeError(f"ASSIGNMENT cell. attribName={attribName} is illegal. "
                                  "attribName is read only variable")
        attribName = property(_CompuCell.CellG_ ## attribName ## _get, set_ ## attribName)
    %}
%enddef

%extend CompuCell3D::CellG{

        CELLG_ATTRIB(volume)
        CELLG_ATTRIB(surface)
        CELLG_ATTRIB(xCM)
        CELLG_ATTRIB(yCM)
        CELLG_ATTRIB(zCM)
        CELLG_ATTRIB(xCOM)
        CELLG_ATTRIB(yCOM)
        CELLG_ATTRIB(zCOM)
        CELLG_ATTRIB(xCOMPrev)
        CELLG_ATTRIB(yCOMPrev)
        CELLG_ATTRIB(zCOMPrev)
        CELLG_ATTRIB(iXX)
        CELLG_ATTRIB(iXY)
        CELLG_ATTRIB(iXZ)
        CELLG_ATTRIB(iYY)
        CELLG_ATTRIB(iYZ)
        CELLG_ATTRIB(iZZ)
        CELLG_ATTRIB(lX)
        CELLG_ATTRIB(lY)
        CELLG_ATTRIB(lZ)
        CELLG_ATTRIB(ecc)
        CELLG_ATTRIB(id)
        CELLG_ATTRIB(clusterId)
        CELLG_ATTRIB(extraAttribPtr)
        CELLG_ATTRIB(pyAttrib)
        %pythoncode %{
        #simplifying access to cell Python dictionary
        def setdict(self,_dict):
            raise AttributeError(f'ASSIGNMENT cell.dict={_dict} is illegal. Dictionary "dict" can only be modified but not replaced')

        def getdict(self):
            dict_object = _CompuCell.getPyAttrib(self)
            return _CompuCell.getPyAttrib(self)
        dict = property(getdict, setdict)

        # simplifying access to sbml models
        def setsbml(self, sbml) :
            raise AttributeError(f'ASSIGNMENT cell.sbml = {sbml} is illegal. '
                                 f'"sbml" attribute can only be modified but not replaced')

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

        sbml = property(getsbml, setsbml)

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

        maboss = property(_get_maboss, _set_maboss)

        %} // pythoncode matching brace




}





    



%include "Field3D/Field3D.h"
%include "Field3D/Field3DImpl.h"
%include "Field3D/WatchableField3D.h"


%include <NeighborFinderParams.h>


%include <CompuCell3D/PluginManager.h>

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


%define FIELD3DEXTENDERBASE(className,returnType)
%extend  className{

        std::string __str__(){
            std::ostringstream s;
            s <<#className << " dim" << self->getDim();
            return s.str();
        }

        returnType min(){
            returnType minVal = self->get(Point3D(0, 0, 0));

            Dim3D dim = self->getDim();

            for (int x = 0; x < dim.x; ++x)
                for (int y = 0; y < dim.y; ++y)
                    for (int z = 0; z < dim.z; ++z) {
                        returnType val = self->get(Point3D(x, y, z));
                        if (val < minVal) minVal = val;
                    }

            return minVal;

        }

        returnType max(){
            returnType maxVal = self->get(Point3D(0, 0, 0));

            Dim3D dim = self->getDim();

            for (int x = 0; x < dim.x; ++x)
                for (int y = 0; y < dim.y; ++y)
                    for (int z = 0; z < dim.z; ++z) {
                        returnType val = self->get(Point3D(x, y, z));
                        if (val > maxVal) maxVal = val;
                    }

            return maxVal;

        }

        returnType __getitem__(PyObject *_indexTuple) {
            if (!PyTuple_Check(_indexTuple) || PyTuple_GET_SIZE(_indexTuple) != 3) {
                throw
                std::runtime_error(std::string(#className)+std::string(
                        ": Wrong Syntax: Expected something like: field[1,2,3]"));
            }
            PyObject *xCoord = PyTuple_GetItem(_indexTuple, 0);
            PyObject *yCoord = PyTuple_GetItem(_indexTuple, 1);
            PyObject *zCoord = PyTuple_GetItem(_indexTuple, 2);
            Py_ssize_t x, y, z;

            //x-coord
            if (PyInt_Check(xCoord)) {
                x = PyInt_AsLong(PyTuple_GetItem(_indexTuple, 0));
            } else if (PyLong_Check(xCoord)) {
                x = PyLong_AsLong(PyTuple_GetItem(_indexTuple, 0));
            } else if (PyFloat_Check(xCoord)) {
                x = (Py_ssize_t) round(PyFloat_AsDouble(PyTuple_GetItem(_indexTuple, 0)));
            } else {
                throw
                std::runtime_error(
                        "Wrong Type (X): only integer or float values are allowed here - floats are rounded");
            }
            //y-coord
            if (PyInt_Check(yCoord)) {
                y = PyInt_AsLong(PyTuple_GetItem(_indexTuple, 1));
            } else if (PyLong_Check(yCoord)) {
                y = PyLong_AsLong(PyTuple_GetItem(_indexTuple, 1));
            } else if (PyFloat_Check(yCoord)) {
                y = (Py_ssize_t) round(PyFloat_AsDouble(PyTuple_GetItem(_indexTuple, 1)));
            } else {
                throw
                std::runtime_error(
                        "Wrong Type (Y): only integer or float values are allowed here - floats are rounded");
            }
            //z-coord
            if (PyInt_Check(zCoord)) {
                z = PyInt_AsLong(PyTuple_GetItem(_indexTuple, 2));
            } else if (PyLong_Check(zCoord)) {
                z = PyLong_AsLong(PyTuple_GetItem(_indexTuple, 2));
            } else if (PyFloat_Check(zCoord)) {
                z = (Py_ssize_t) round(PyFloat_AsDouble(PyTuple_GetItem(_indexTuple, 2)));
            } else {
                throw
                std::runtime_error(
                        "Wrong Type (Z): only integer or float values are allowed here - floats are rounded");
            }

            return self->get(Point3D(x, y, z));
        }
}

%enddef


%define CELLFIELD3DEXTENDER(className, returnType)
FIELD3DEXTENDERBASE(className,returnType)
%extend className {
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
            throw std::runtime_error("Wrong Syntax: Expected something like: field[1,2,3]=object");
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


%define FIELD3DEXTENDER(className,returnType)
FIELD3DEXTENDERBASE(className,returnType)

%extend className{

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
        throw std::runtime_error("Wrong Syntax: Expected something like: field[1,2,3]=object");
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
CELLFIELD3DEXTENDER(CompuCell3D::Field3D<CompuCell3D::CellG *>,CompuCell3D::CellG*)
FIELD3DEXTENDER(CompuCell3D::Field3D<float>,float)
FIELD3DEXTENDER(CompuCell3D::Field3D<int>,int)



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

	if(Simulator::steppableManager.isLoaded("ReactionDiffusionSolverFVM")) {
		ReactionDiffusionSolverFVM * solver = (ReactionDiffusionSolverFVM *)Simulator::steppableManager.get("ReactionDiffusionSolverFVM");
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
//    cerr<< "initialize plugin fcn"<<endl;
    CC3D_Log(LOG_DEBUG) << "initialize plugin fcn";

    char *steppablePath = getenv("COMPUCELL3D_STEPPABLE_PATH");

//    cerr<<"steppablePath=" << steppablePath<<endl;

    CC3D_Log(LOG_DEBUG) << "steppablePath=" << steppablePath;

    if (steppablePath)
        Simulator::steppableManager.loadLibraries(steppablePath);
//	cerr<<"after Simulator::steppableManager.loadLibraries(steppablePath)"<<endl;
    char *pluginPath = getenv("COMPUCELL3D_PLUGIN_PATH");
    CC3D_Log(LOG_DEBUG) << "pluginPath=" << pluginPath;
//    cerr << "pluginPath=" << pluginPath<<endl;
    CC3D_Log(LOG_DEBUG) << "THIS IS JUST BEFORE LOADING LIBRARIES";
      
   
    if (pluginPath)
        Simulator::pluginManager.loadLibraries(pluginPath);

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
      CC3D_Log(LOG_DEBUG) << "ModuleName=" << source->moduleName;
   }

%}




%include "CompuCellExtraDeclarations.i"

%include "DerivedProperty.i"

%rename(_usePermeableSurfaces) ReactionDiffusionSolverFVM::usePermeableSurfaces(unsigned int&);
%include "CompuCell3D/steppables/PDESolvers/ReactionDiffusionSolverFVM.h"

%inline %{

 ReactionDiffusionSolverFVM * getReactionDiffusionSolverFVMSteppable(){

      return (ReactionDiffusionSolverFVM *)Simulator::steppableManager.get("ReactionDiffusionSolverFVM");

   }

%}
