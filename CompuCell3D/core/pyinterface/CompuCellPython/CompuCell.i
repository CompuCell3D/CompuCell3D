
// -*-c++-*-

// CompuCell.y



// This SWIG interface file defines the CompuCell API.
// CompuCell module wraps the major CompuCell3D classes.

// Module Name


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


// CompuCell Include Files

// CompuCell3D Include Files
#include <CompuCell3D/Field3D/Neighbor.h>
#include <CompuCell3D/Boundary/BoundaryStrategy.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <CompuCell3D/ClassRegistry.h>
#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>
#include <CompuCell3D/Plugin.h>
#include <CompuCell3D/Steppable.h>
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
// #include <BasicUtils/BasicPluginManager.h>

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

// ************************************************************
// SWIG Declarations
// ************************************************************

// Where possible, classes are presneted to SWIG via %include.
// SWIG simply uses the definition in the include file and builds
// wrappers based on it.

// In a few cases (e.g. Field3D), SWIG became confused and could not 
// properly handle the header file.  These classes and support definitions
// are explicitly handled here.

// Additionally, the definitions for some of the third party classes
// are explicit here.  This may change in the future.

// ******************************
// SWIG Libraries
// ******************************

// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"

// C++ std::map handling
%include "std_set.i"

// C++ std::map handling
%include "std_vector.i"

// ******************************
// Third Party Classes
// ******************************

%include "BasicUtils/BasicClassFactoryBase.h"
%include "BasicUtils/BasicClassFactory.h"
// %include <CompuCell3D/Plugin.h>
// %include <BasicUtils/BasicPluginManager.h>

template <class T>
class BasicPluginManager {
  public:
    void loadLibraries(const std::string path);
    bool isLoaded(std::string pluginName);
};

%template(templatebasicpluginmanagerplugin) BasicPluginManager<Plugin>;


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
#define CONTACTLOCALFLEX_EXPORT
#define CONTACTLOCALPRODUCT_EXPORT
#define CONTACTMULTICAD_EXPORT
#define CELLORIENTATION_EXPORT
#define POLARIZATIONVECTOR_EXPORT
#define ELASTICITY_EXPORT
#define PLASTICITY_EXPORT
#define CONNECTIVITYLOCALFLEX_EXPORT
// #define LENGTHCONSTRAINTLOCALFLEX_EXPORT
#define LENGTHCONSTRAINT_EXPORT
#define MOLECULARCONTACT_EXPORT 
#define SECRETION_EXPORT 
#define VOLUMETRACKER_EXPORT 

//AutogeneratedModules - DO NOT REMOVE THIS LINE IT IS USED BY TWEDIT TO LOCATE CODE INSERTION POINT
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


%include <CompuCell3D/Field3D/Neighbor.h>
%include <CompuCell3D/Boundary/BoundaryStrategy.h>
%include "Potts3D/Cell.h"




%include "Field3D/Point3D.h"
%include "Field3D/Dim3D.h"
%include "Field3D/WatchableField3D.h"

%include <NeighborFinderParams.h>


%include <CompuCell3D/PluginManager.h>
%template(pluginmanagertemplate) CompuCell3D::PluginManager<Plugin> ;
%template(steppablemanagertemplate) CompuCell3D::PluginManager<Steppable> ;



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



// The template instanciation for Field3D<Cell *> does not work properly
// if Field3D.h is included and the template is instanciated by SWIG.
// However, simply declaring Field3D and instanciating Field3D<Cell *>
// this way seems to work fine.
namespace CompuCell3D {
  template <class T>
  class Field3D {
  public:
    virtual void set(const Point3D &pt, const T value) = 0;
    virtual T get(const Point3D &pt) const = 0;
    // virtual T operator[](const Point3D &pt) const {return get(pt);}
    // --> Warning(389): operator[] ignored (consider using %extend)
    virtual Dim3D getDim() const = 0;
    virtual bool isValid(const Point3D &pt) const = 0;
    virtual void setDim(const Dim3D theDim) {}
    virtual Point3D getNeighbor(const Point3D &pt, unsigned int &token,
				double &distance,
    			bool checkBounds = true) const;
    %extend  {
      T get(short x, short y, short z) {
        Point3D pt;
        pt.x = x; pt.y = y; pt.z = z;
        return self->get(pt);
      };


		Point3D nextNeighbor(NeighborFinderParams & nfp){
			Point3D n=self->getNeighbor(nfp.pt , nfp.token , nfp.distance, nfp.checkBounds);
			return n;
		}
      double producePoint(double seed,Point3D & result){
         result.x=3;
         result.y=8;
         result.z=6;
         return 4;
      };


      void produceNumber(double seed,double & result){
         result=88;
  
      };


    };
  }; 


  %template(cellfield) Field3D<CellG *>;
  %template(floatfield) Field3D<float>;
  %template(floatfieldImpl) Field3DImpl<float>;
  %template(watchablecellfield) WatchableField3D<CellG *>;
  

  
  //%template (simulatorFieldMap) ::std::map<std::string,Field3D<float> >;
  //%template (simulatorFieldMap) ::std::set<int>;
};

%template(vectorstdstring) std::vector<std::string>;
%template(vectordouble) std::vector<double>;
%template(vectorint) std::vector<int>;


//%template (simulatorFieldMap) ::std::map<std::string,int >;

//%template (simulatorFieldMap) std::map<std::string,CompuCell3D::Field3DImpl<float>*>;

%extend CompuCell3D::Point3D{
   char * __str__(){
      static char id[100];
      sprintf(id,"(%d,%d,%d)",self->x,self->y,self->z);
      return id;
   }
   double produceNumber(double seed,double & result){
         result=88;
         return 4;
   };
   
    %insert("python") %{
        def __getstate__(self):
            return (self.x,self.y,self.z)
            
        def __setstate__(self,tup):
            print 'tuple=',tup
            self.this = _CompuCell.new_Point3D(tup[0],tup[1],tup[2])
            self.thisown=1            
            
    %}   

    
};


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


%include <CompuCell3D/ParseData.h>
%include <CompuCell3D/ParserStorage.h>
%include <CompuCell3D/PottsParseData.h>

%include <PublicUtilities/NumericalUtils.h>
%include <PublicUtilities/Vector3.h>

%include <BasicUtils/BasicException.h>


%include exception.i


// ************************************************************
// Init Functions
// ************************************************************


// ************************************************************
// Inline Functions
// ************************************************************


%inline %{

   Field3D<float> * getConcentrationField(CompuCell3D::Simulator & simulator, std::string fieldName){
      std::map<std::string,Field3DImpl<float>*> & fieldMap=simulator.getConcentrationFieldNameMap();
      std::map<std::string,Field3DImpl<float>*>::iterator mitr;
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
