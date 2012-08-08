

%module ("threads"=1) BionetSolverPy
// %module BionetSolverPy
#define SwigPyIterator BionetSolverPy_SwigPyIterator
%{
#define SwigPyIterator BionetSolverPy_SwigPyIterator
%}
// %rename (BionetSolverPy_SwigPyIterator) SwigPyIterator;

%include "typemaps.i"

%include <windows.i>

%{

#include "bionetsolver/soslib_OdeModel.h"
#include "bionetsolver/soslib_CvodeSettings.h"
#include "bionetsolver/soslib_IntegratorInstance.h"

#include "bionetsolver/BionetworkSBML.h"
#include "bionetsolver/BionetworkTemplateLibrary.h"
#include "bionetsolver/Bionetwork.h"
#include "bionetsolver/BionetworkUtilManager.h"

#include <iostream>

using namespace std;

%}
// have to ignore SwigPyIterator or else there is a conflict with SwigPyIterator from CC3D modules
// it is quite difficult to rteacwe back what exactly causes this problem
// for now the solution is to ignore generation of SwigPyIterator\

//here is a reference found on the web to the bug in Swig
// # 1. Workaround for SWIG bug #1863647: Ensure that the PySwigIterator class
// #    (SwigPyIterator in 1.3.38 or later) is renamed with a module-specific
// #    prefix, to avoid collisions when using multiple modules
// # 2. If module names contain '.' characters, SWIG emits these into the CPP
// #    macros used in the director header. Work around this by replacing them
// #    with '_'. A longer term fix is not to call our modules "IMP.foo" but
// #    to say %module(package=IMP) foo but this doesn't work in SWIG stable
// #    as of 1.3.36 (Python imports incorrectly come out as 'import foo'
// #    rather than 'import IMP.foo'). See also IMP bug #41 at
// #    https://salilab.org/imp/bugs/show_bug.cgi?id=41

// %ignore SwigPyIterator;

// %rename (SwigPyIteratorBionetSolverPy) SwigPyIterator;

// SOLUTION 1 WOULD BE

// %rename (BionetSolverPySwigPyIterator) SwigPyIterator;

// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"
namespace std {
    typedef map<unsigned int, double> UIntDoubleMap;
    %template(UIntDoubleMap) map<unsigned int, double>;
}

// C++ std::pair handling
%include "std_pair.i"
namespace std {
    typedef pair<bool, double> BoolDoublePair;
    %template(BoolDoublePair) pair<bool, double>;
}

// C++ std::map handling
%include "std_set.i"

// C++ std::vector handling
%include "std_vector.i"




#define BIONETWORK_EXPORT

%ignore BionetworkTemplateLibrary::setInitialCondition(variableType, std::string, double);
%ignore Bionetwork::initializeIntegrators(std::map<std::string, const BionetworkSBML *>);
%ignore Bionetwork::printIntracellularState();

%include "bionetsolver/soslib_OdeModel.h"
%include "bionetsolver/soslib_CvodeSettings.h"
%include "bionetsolver/soslib_IntegratorInstance.h"

%include "bionetsolver/BionetworkSBML.h"
%include "bionetsolver/BionetworkTemplateLibrary.h"
%include "bionetsolver/Bionetwork.h"
%include "bionetsolver/BionetworkUtilManager.h"





