
// Module Name
%module("threads"=1) RoadRunner

//%module Example
// ************************************************************
// Module Includes 
// ************************************************************

// These are copied directly to the .cxx file and are not parsed
// by SWIG.  Include include files or definitions that are required
// for the module to build correctly.

// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"

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



// %include "Fields_pre.i"

%{
    #include <numpy/arrayobject.h>
    #include <lsComplex.h>
    #include <lsMatrix.h>
    #include <lsLibla.h>
    #include <lsLA.h>
    #include <lsLUResult.h>    
    #include <lsUtils.h>
    // #include <rrStringList.h>
    #include <rrRoadRunner.h>    
    using namespace std;
    using namespace ls;
    
    using namespace rr;

%}


%include "swig_includes/numpy.i"

%init %{
    import_array();
%}

#define LIB_EXTERN
#define RR_DECLSPEC
#define PUGIXML_CLASS


%include <lsComplex.h>

%include <lsMatrix.h>
%template (DoubleMatrixTemplate) ls::Matrix<double>;
%template (ComplexMatrixTemplate) ls::Matrix<ls::Complex>;
%template (IntMatrixTemplate) ls::Matrix<int>;


%include <lsLUResult.h>
%include <lsUtils.h>



// %include <rrStringList.h>
// %include <lsLibla.h>
// %include <lsLA.h>
//have to ignore API fcns which have no implementation to avod link errors

%ignore rr::RoadRunner::computeContinuation;
%ignore rr::RoadRunner::getSpeciesIds;
%ignore rr::RoadRunner::steadyStateParameterScan;
%ignore rr::RoadRunner::setLocalParameterByIndex;
%ignore rr::RoadRunner::setLocalParameterValues;
%ignore rr::RoadRunner::getLocalParameterIds;
%ignore rr::RoadRunner::getAllLocalParameterTupleList;
%ignore rr::RoadRunner::setCompartmentVolumes;
%ignore rr::RoadRunner::setGlobalParameterValues;
%ignore rr::RoadRunner::getAllGlobalParameterTupleList;
%ignore rr::RoadRunner::getName;
%ignore rr::RoadRunner::getAuthor;
%ignore rr::RoadRunner::getDisplayName;
%ignore rr::RoadRunner::loadPlugins;
%ignore rr::RoadRunner::unLoadPlugins;
%ignore rr::RoadRunner::getUnScaledElasticity;
%ignore rr::RoadRunner::getUnscaledFloatingSpeciesElasticity;
%ignore rr::RoadRunner::changeParameter;
%ignore rr::RoadRunner::getUnscaledParameterElasticity;
%ignore rr::RoadRunner::getUnscaledConcentrationControlCoefficient;
%ignore rr::RoadRunner::getScaledConcentrationControlCoefficient;
%ignore rr::RoadRunner::getUnscaledConcentrationControlCoefficient;
%ignore rr::RoadRunner::getScaledConcentrationControlCoefficient;
%ignore rr::RoadRunner::getUnscaledFluxControlCoefficient;
%ignore rr::RoadRunner::getUnscaledFluxControlCoefficient;
%ignore rr::RoadRunner::getScaledFluxControlCoefficient;
%ignore rr::RoadRunner::getScaledFluxControlCoefficient;
%ignore rr::RoadRunner::computeSteadyStateValues;
%ignore rr::RoadRunner::getSelectedValues;
%ignore rr::RoadRunner::getLocalParameterValues;
%ignore rr::RoadRunner::getFloatingSpeciesIdsArray;
%ignore rr::RoadRunner::getGlobalParameterIdsArray;



%template (mapStringDouble) std::map<std::string, double>;

%include <rrRoadRunner.h>

