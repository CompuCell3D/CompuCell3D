#ifndef rrConstantsH
#define rrConstantsH
#include <limits>
#include <string>
#include "rrExporter.h"
using namespace std;

namespace rr
{

//Useful constants...
RR_DECLSPEC extern const char 	gPathSeparator;
//RR_DECLSPEC extern const string	gEmptyString="";
RR_DECLSPEC extern const string	gEmptyString;
RR_DECLSPEC extern const string gExeSuffix;

RR_DECLSPEC extern const char* 	gDoubleFormat;
RR_DECLSPEC extern const char* 	gIntFormat;
RR_DECLSPEC extern const char* 	gComma;
RR_DECLSPEC extern const string gDefaultSupportCodeFolder;
RR_DECLSPEC extern const string gDefaultCompiler;
RR_DECLSPEC extern const string gDefaultTempFolder;

//Messages
RR_DECLSPEC extern const string gEmptyModelMessage;


// Constants
RR_DECLSPEC extern const char 	gTab;
RR_DECLSPEC extern const char 	gNL;
RR_DECLSPEC extern const double	gDoubleNaN;
RR_DECLSPEC extern const float  gFloatNaN;
RR_DECLSPEC extern const int    gMaxPath;

// Enums...
enum SBMLType {stCompartment = 0, stSpecies, stParameter};    //Species clashes with class Species, prefix enums with st, for SbmlType

// Typedefs
typedef unsigned int 			u_int;


}
#endif
