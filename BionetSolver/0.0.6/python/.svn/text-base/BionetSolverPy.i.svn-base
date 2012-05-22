

%module ("threads"=1) BionetSolverPy

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





