#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include "rrStringUtils.h"
#include "rrConstants.h"

//---------------------------------------------------------------------------
namespace rr
{

//Useful constants..
const char*     	gComma 			            = ",";
const char 	 		gTab 			            = '\t';
const char 	 		gNL 			            = '\n';
const char* 		gDoubleFormat 	            = "%f";
const char* 		gIntFormat  	            = "%d";;

const string    	gEmptyString 				= "";
//gEmptyString 				= "";

//Observe, the following functions are executed BEFORE any main..
const string		gDefaultSupportCodeFolder 	= JoinPath("..", "rr_support");
const string		gDefaultTempFolder 			= ".";

const int 			gMaxPath					= 512;
const double    	gDoubleNaN   				= std::numeric_limits<double>::quiet_NaN() ;
const float     	gFloatNaN    				= std::numeric_limits<float>::quiet_NaN() ;

//Messages
const string		gEmptyModelMessage 			= "A model needs to be loaded before one can use this method";


#if defined(_WIN32) || defined(__CODEGEARC__)
const string		gDefaultCompiler 			= JoinPath("..", "compilers", "tcc", "tcc.exe");
const char       	gPathSeparator      = '\\';
const string		gExeSuffix          = ".exe";
#elif defined(__linux)
const string		gDefaultCompiler    = "gcc";
const char       	gPathSeparator      = '/';
const string		gExeSuffix          = "";
#else  //Something else...
const char       	gPathSeparator      = '/';
const string		gExeSuffix          = "";
#endif

}

