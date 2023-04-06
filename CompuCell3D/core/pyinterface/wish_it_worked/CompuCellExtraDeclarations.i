

// ************************************************************
// SWIG Declarations
// ************************************************************


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



// C++ std::vector handling
%include "std_vector.i"


%include "Potts3D/Cell.h"
%include "Field3D/Point3D.h"
%include "Field3D/Dim3D.h"
%include <NeighborFinderParams.h>


using namespace CompuCell3D;

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


      /*T getWithNullCheck(Point3D pt) {
         T  cell;
         cell=self->get(pt);
         if(!cell)
            cerr<<"GOT NULL PTR TO CELL"<<endl;
        return cell;
      };*/

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
};





//adding cout capabilities to Point3D
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
};


//%include "Field3D/Field3DChangeWatcher.h"
//%template(cellchangewatcher) CompuCell3D::Field3DChangeWatcher<CompuCell3D::Cell *>;

//%include "Potts3D/CellChangeWatcher.h"
%include <CompuCell3D/Automaton/Automaton.h>
%include <CompuCell3D/Potts3D/Potts3D.h>
%include "Steppable.h"
%include "ClassRegistry.h"
%include "Simulator.h"

%include <Utils/Coordinates3D.h>

%template (Coordinates3DDouble) Coordinates3D<double>; 

%include <PyCompuCellObjAdapter.h>
%include <EnergyFunctionPyWrapper.h>
%include <ChangeWatcherPyWrapper.h>
%include <TypeChangeWatcherPyWrapper.h>
%include <StepperPyWrapper.h>

// %nothread PyAttributeAdder;
%include <PyAttributeAdder.h>

// %nothreadblock %{
    // class PyAttributeAdder :public PyCompuCellObjAdapter , public AttributeAdder{
        // public:
          // PyAttributeAdder():refChecker(0),destroyer(0){}
          // virtual void addAttribute(CellG *);
          // virtual void destroyAttribute(CellG *);
          // AttributeAdder * getPyAttributeAdderPtr();
          // void registerAdder(PyObject *);
          // void registerRefChecker(PyObject * _refChecker){refChecker=_refChecker;}
          // void registerDestroyer(PyObject * _destroyer){destroyer=_destroyer;}

          // PyObject * refChecker;
          // PyObject * destroyer;
    // }
// %}
// %clearnothreadblock



%include "ParseData.h"
%include "ParserStorage.h"

%template (VectorParseDataPtr) std::vector<ParseData*> ;


//have to include all  export definitions for modules which are arapped to avoid problems with interpreting by swig win32 specific c++ extensions...
#define COMPUCELLLIB_EXPORT
#define LOGGER_EXPORT
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
#define LENGTHCONSTRAINTLOCALFLEX_EXPORT
#define MOLECULARCONTACT_EXPORT 
#define FOCALPOINTPLASTICITY_EXPORT
#define MOMENTOFINERTIA_EXPORT 
#define ADHESIONFLEX_EXPORT



%inline %{
   PyObject * getPyAttrib(CompuCell3D::CellG * _cell){
      Py_INCREF(_cell->pyAttrib);
      return _cell->pyAttrib;
   }
%}

//Plugins

%include <BasicUtils/BasicClassAccessor.h>
%include <BasicUtils/BasicClassGroup.h> //had to include it to avoid problems with template instantiation


%inline %{
   Plugin * getPlugin(std::string _pluginName){
      return (Plugin *)Simulator::pluginManager.get(_pluginName);
   }

   Steppable * getSteppable(std::string _steppableName){
      return (Steppable *)Simulator::steppableManager.get(_steppableName);
   }


%}

//ConnectivityLocalFlex
%include <CompuCell3D/plugins/ConnectivityLocalFlex/ConnectivityLocalFlexData.h>
%template (connectivitylocalflexccessor) BasicClassAccessor<ConnectivityLocalFlexData>; //necessary to get ConnectivityLocalFlexData accessor working

%include <CompuCell3D/plugins/ConnectivityLocalFlex/ConnectivityLocalFlexPlugin.h>

%inline %{
   ConnectivityLocalFlexPlugin * getConnectivityLocalFlexPlugin(){
         return (ConnectivityLocalFlexPlugin *)Simulator::pluginManager.get("ConnectivityLocalFlex");
    }
%}

//LengthConstraintLocalFlex
%include <CompuCell3D/plugins/LengthConstraintLocalFlex/LengthConstraintLocalFlexData.h>
%template (lengthconstraintlocalflexccessor) BasicClassAccessor<LengthConstraintLocalFlexData>; //necessary to get LengthConstraintLocalFlexData accessor working

%include <CompuCell3D/plugins/LengthConstraintLocalFlex/LengthConstraintLocalFlexPlugin.h>

%inline %{
   LengthConstraintLocalFlexPlugin * getLengthConstraintLocalFlexPlugin(){
         return (LengthConstraintLocalFlexPlugin *)Simulator::pluginManager.get("LengthConstraintLocalFlex");
    }
%}

%include <CompuCell3D/plugins/ChemotaxisSimple/ChemotaxisSimpleEnergy.h>

//Chemotaxis Plugin
%include <CompuCell3D/plugins/Chemotaxis/ChemotaxisData.h>
%include <CompuCell3D/plugins/Chemotaxis/ChemotaxisPlugin.h>

%inline %{
   ChemotaxisPlugin * getChemotaxisPlugin(){
      return (ChemotaxisPlugin *)Simulator::pluginManager.get("Chemotaxis");
   }

%}



//plugins
// %include <CompuCell3D/plugins/Mitosis/MitosisParseData.h>
%include <CompuCell3D/plugins/Mitosis/MitosisPlugin.h>
%include <CompuCell3D/plugins/Mitosis/MitosisSimplePlugin.h>



//NeighborPlugin

%include <CompuCell3D/plugins/NeighborTracker/NeighborTracker.h>

%template (neighbortrackeraccessor) BasicClassAccessor<NeighborTracker>; //necessary to get NeighborTracker accessor working

// %template (nsdSetPyItr) STLPyIterator<std::set<CompuCell3D::NeighborSurfaceData> >;
%template (neighborsurfacedataset) std::set<CompuCell3D::NeighborSurfaceData>; //necessary to get basis set functionality working


%include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>


// %inline %{
//    PyObject * getPyAttrib(CompuCell3D::CellG * _cell){
//       Py_INCREF(_cell->pyAttrib);
//       return _cell->pyAttrib;
//    }
// %}

%inline %{
   NeighborTrackerPlugin * getNeighborTrackerPlugin(){
      return (NeighborTrackerPlugin *)Simulator::pluginManager.get("NeighborTracker");
   }

   CompuCell3D::NeighborSurfaceData & derefNeighborSurfaceData(std::set<CompuCell3D::NeighborSurfaceData>::iterator &_itr){
      return const_cast<CompuCell3D::NeighborSurfaceData &>(*_itr);
   }

%}

%include <CompuCell3D/plugins/PixelTracker/PixelTracker.h>

%template (PixelTrackerAccessor) BasicClassAccessor<PixelTracker>; //necessary to get PixelTracker accessor working

// %template (pixelSetPyItr) STLPyIterator<std::set<CompuCell3D::PixelTrackerData> >;
%template (PixelTrackerDataset) std::set<CompuCell3D::PixelTrackerData>; //necessary to get basis set functionality working

%include <CompuCell3D/plugins/PixelTracker/PixelTrackerPlugin.h>

%inline %{
   PixelTrackerPlugin * getPixelTrackerPlugin(){
      return (PixelTrackerPlugin *)Simulator::pluginManager.get("PixelTracker");
   }

%}

%include <CompuCell3D/plugins/PixelTracker/BoundaryPixelTracker.h>
%template (BoundaryPixelTrackerAccessor) BasicClassAccessor<BoundaryPixelTracker>; //necessary to get BoundaryPixelTracker accessor working

// %template (boundaryPixelSetPyItr) STLPyIterator<std::set<CompuCell3D::BoundaryPixelTrackerData> >;
%template (BoundaryPixelTrackerDataset) std::set<CompuCell3D::BoundaryPixelTrackerData>; //necessary to get basis set functionality working

%include <CompuCell3D/plugins/PixelTracker/BoundaryPixelTrackerPlugin.h>

%inline %{
   BoundaryPixelTrackerPlugin * getBoundaryPixelTrackerPlugin(){
      return (BoundaryPixelTrackerPlugin *)Simulator::pluginManager.get("BoundaryPixelTracker");
   }


%}

//ContactLocalFlexPlugin

%include <CompuCell3D/plugins/ContactLocalFlex/ContactLocalFlexData.h>
%template (contactlocalflexcontainerccessor) BasicClassAccessor<ContactLocalFlexDataContainer>; //necessary to get ContactlocalFlexData accessor working

// %template (clfdSetPyItr) STLPyIterator<std::set<CompuCell3D::ContactLocalFlexData> >;
%template (contactlocalflexdataset) std::set<CompuCell3D::ContactLocalFlexData>; //necessary to get basis set functionality working

%include <CompuCell3D/plugins/ContactLocalFlex/ContactLocalFlexPlugin.h>

%inline %{
   ContactLocalFlexPlugin * getContactLocalFlexPlugin(){
      return (ContactLocalFlexPlugin *)Simulator::pluginManager.get("ContactLocalFlex");
   }
%}

//ContactLocalProductPlugin
%include <CompuCell3D/plugins/ContactLocalProduct/ContactLocalProductData.h>
%template (contactproductflexccessor) BasicClassAccessor<ContactLocalProductData>; //necessary to get ContactLocalProductData accessor working

// %template (jVecPyItr) STLPyIterator<ContactLocalProductData::ContainerType_t>; //ContainerType_t - this is vector<float> in current implementation
%template (contactproductdatacontainertype) std::vector<float>; //necessary to get basis vector functionality working


//some functions to get more vector function to work

%extend std::vector<float>{

   void set(unsigned int pos, float _x){
      if(pos <= self->size()-1 ){
         self->operator[](pos)=_x;
      }
   }

   float get(unsigned int pos){
      if(pos <= self->size()-1 ){
         return self->operator[](pos);
      }
   }


}

%include <CompuCell3D/plugins/ContactLocalProduct/ContactLocalProductPlugin.h>

%inline %{
   ContactLocalProductPlugin * getContactLocalProductPlugin(){
      return (ContactLocalProductPlugin *)Simulator::pluginManager.get("ContactLocalProduct");
   }
%}

//ContactMultiCadPlugin
%include <CompuCell3D/plugins/ContactMultiCad/ContactMultiCadData.h>
%template (contactmulticaddataaccessor) BasicClassAccessor<ContactMultiCadData>; //necessary to get ContactMultiCadData accessor working

%include <CompuCell3D/plugins/ContactMultiCad/ContactMultiCadPlugin.h>

%inline %{
   ContactMultiCadPlugin * getContactMultiCadPlugin(){
      return (ContactMultiCadPlugin *)Simulator::pluginManager.get("ContactMultiCad");
   }
%}

//AdhesionFlexPlugin
%include <CompuCell3D/plugins/AdhesionFlex/AdhesionFlexData.h>
%template (adhesionflexdataaccessor) BasicClassAccessor<AdhesionFlexData>; //necessary to get AdhesionFlexData accessor working

%include <CompuCell3D/plugins/AdhesionFlex/AdhesionFlexPlugin.h>

%inline %{
   AdhesionFlexPlugin * getAdhesionFlexPlugin(){
      return (AdhesionFlexPlugin *)Simulator::pluginManager.get("AdhesionFlex");
   }
%}



//CellOrientation Plugin
%include <CompuCell3D/plugins/CellOrientation/CellOrientationVector.h>
%template (cellOrientationVectorAccessor) BasicClassAccessor<CellOrientationVector>; //necessary to get CellOrientationVector accessor working

%template (LambdaCellOrientationAccessor) BasicClassAccessor<LambdaCellOrientation>; //necessary to get LambdaCellOrientation accessor working

%include <CompuCell3D/plugins/CellOrientation/CellOrientationPlugin.h>

%inline %{
   CellOrientationPlugin * getCellOrientationPlugin(){
      return (CellOrientationPlugin *)Simulator::pluginManager.get("CellOrientation");
   }

%}

//PolarizationVectorPlugin
%include <CompuCell3D/plugins/PolarizationVector/PolarizationVector.h>
%template (polarizationVectorAccessor) BasicClassAccessor<PolarizationVector>; //necessary to get CellOrientationVector accessor working

%include <CompuCell3D/plugins/PolarizationVector/PolarizationVectorPlugin.h>

%inline %{
   PolarizationVectorPlugin * getPolarizationVectorPlugin(){
      return (PolarizationVectorPlugin*)Simulator::pluginManager.get("PolarizationVector");
   }

%}

//Elasticity Plugin
%include <CompuCell3D/plugins/Elasticity/ElasticityTracker.h>
%template (elasticityTrackerAccessor) BasicClassAccessor<ElasticityTracker>; //necessary to get ElasticityTracker accessor working

// %template (elasticitySetPyItr) STLPyIterator<std::set<CompuCell3D::ElasticityTrackerData> >;
%template (elasticityTrackerDataSet) std::set<CompuCell3D::ElasticityTrackerData>; //necessary to get basic set functionality working


%include <CompuCell3D/plugins/Elasticity/ElasticityTrackerPlugin.h>

%inline %{
 ElasticityTrackerPlugin * getElasticityTrackerPlugin(){
         return (ElasticityTrackerPlugin *)Simulator::pluginManager.get("ElasticityTracker");
   }

 CompuCell3D::ElasticityTrackerData & derefElasticityTrackerData(std::set<CompuCell3D::ElasticityTrackerData>::iterator &_itr){
     return const_cast<CompuCell3D::ElasticityTrackerData &>(*_itr);
 }


%}


//Plasticity Plugin
%include <CompuCell3D/plugins/Plasticity/PlasticityTracker.h>
%template (plasticityTrackerAccessor) BasicClassAccessor<PlasticityTracker>; //necessary to get PlasticityTracker accessor working

// %template (plasticitySetPyItr) STLPyIterator<std::set<CompuCell3D::PlasticityTrackerData> >;
%template (plasticityTrackerDataSet) std::set<CompuCell3D::PlasticityTrackerData>; //necessary to get basic set functionality working


%include <CompuCell3D/plugins/Plasticity/PlasticityTrackerPlugin.h>

%inline %{
 PlasticityTrackerPlugin * getPlasticityTrackerPlugin(){
      return (PlasticityTrackerPlugin *)Simulator::pluginManager.get("PlasticityTracker");
   }

CompuCell3D::PlasticityTrackerData & derefPlasticityTrackerData(std::set<CompuCell3D::PlasticityTrackerData>::iterator &_itr){
      return const_cast<CompuCell3D::PlasticityTrackerData &>(*_itr);
   }



%}


//Focal Point Plasticity Plugin
%include <CompuCell3D/plugins/FocalPointPlasticity/FocalPointPlasticityTracker.h>
%template (focalPointPlasticityTrackerAccessor) BasicClassAccessor<FocalPointPlasticityTracker>; //necessary to get PlasticityTracker accessor working

// %template (focalPointPlasticitySetPyItr) STLPyIterator<std::set<CompuCell3D::FocalPointPlasticityTrackerData> >;
%template (focalPointPlasticityTrackerDataSet) std::set<CompuCell3D::FocalPointPlasticityTrackerData>; //necessary to get basic set functionality working


%include <CompuCell3D/plugins/FocalPointPlasticity/FocalPointPlasticityPlugin.h>

%inline %{
 FocalPointPlasticityPlugin * getFocalPointPlasticityPlugin(){
      return (FocalPointPlasticityPlugin *)Simulator::pluginManager.get("FocalPointPlasticity");
   }

CompuCell3D::FocalPointPlasticityTrackerData & derefPlasticityTrackerData(std::set<CompuCell3D::FocalPointPlasticityTrackerData>::iterator &_itr){
      return const_cast<CompuCell3D::FocalPointPlasticityTrackerData &>(*_itr);
   }

%}


// //MolecularContactPlugin
// %include <CompuCell3D/plugins/MolecularContact/MolecularContactPlugin.h>
// /* %nothread MolecularContactPlugin */
// /* %template (moleculemapaccessor) BasicClassAccessor<MoleculeNameMap_t>; //necessary to get MolecularContactPlugin accessor working */

// /* %template (molecularlcontactMapPyItr) STLPyIterator<std::map<CompuCell3D::MolecularContactData> >; */
// /* %template (molecularlcontactDataSet) std::map<CompuCell3D::MolecularContactData>; //necessary to get basic map functionality working */

// %inline %{
   // MolecularContactPlugin * getMolecularContactPlugin(){
      // return (MolecularContactPlugin *)Simulator::pluginManager.get("MolecularContact");
   // }
// %}


//MomentOfInertia
%include <CompuCell3D/plugins/MomentOfInertia/MomentOfInertiaPlugin.h>

%inline %{
   MomentOfInertiaPlugin * getMomentOfInertiaPlugin(){
      return (MomentOfInertiaPlugin *)Simulator::pluginManager.get("MomentOfInertia");
   }
%}

//Secretion
%include <CompuCell3D/plugins/Secretion/FieldSecretor.h>
%include <CompuCell3D/plugins/Secretion/SecretionPlugin.h>

%inline %{
   SecretionPlugin * getSecretionPlugin(){
      return (SecretionPlugin *)Simulator::pluginManager.get("Secretion");
   }
%}



//Steppables
%include <CompuCell3D/steppables/Mitosis/MitosisSteppable.h>


// #if defined(SWIGPYTHON)
// %template(pyset) std::set<swig::SwigPtr_PyObject>; 
// #endif

// //List of fields from simulator


// //%template (simulatorFieldMapPyItr) STLPyIterator<std::map<std::string,Field3DImpl<float>*> >;
// //%template (simulatorFieldMap) <std::map<std::string,Field3DImpl<float>*>;


