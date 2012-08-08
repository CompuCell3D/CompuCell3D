

%module ("threads"=1) CompuCellExtraModules

%include "typemaps.i"

%include <windows.i>

%{

#include "ParseData.h"
#include "ParserStorage.h"
#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>

// ********************************************* PUT YOUR PLUGIN PARSE DATA AND PLUGIN FILES HERE *************************************************

#include <SimpleVolume/SimpleVolumePlugin.h>
// #include <SimpleVolume/SimpleVolumeParseData.h>
//
#include <VolumeMean/VolumeMean.h>
// #include <VolumeMean/VolumeMeanParseData.h>


// ********************************************* END OF SECTION ********************************** ************************************************


//have to include all  export definitions for modules which are arapped to avoid problems with interpreting by swig win32 specific c++ extensions...
#define SIMPLEVOLUME_EXPORT
#define VOLUMEMEAN_EXPORT


#include <iostream>

using namespace std;
using namespace CompuCell3D;

%}

// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"

// C++ std::map handling
%include "std_set.i"

// C++ std::vector handling
%include "std_vector.i"


//have to include all  export definitions for modules which are arapped to avoid problems with interpreting by swig win32 specific c++ extensions...
#define SIMPLEVOLUME_EXPORT
#define VOLUMEMEAN_EXPORT


%include "ParseData.h"
%include "ParserStorage.h"


// ********************************************* PUT YOUR PLUGIN PARSE DATA AND PLUGIN FILES HERE *************************************************
// REMEMBER TO CHANGE #include to %include


%include <SimpleVolume/SimpleVolumePlugin.h>
// %include <SimpleVolume/SimpleVolumeParseData.h>


// THIS IS VERY IMORTANT STETEMENT WITHOUT IT SWIG will produce incorrect wrapper code which will compile but will not work
using namespace CompuCell3D;



%inline %{
   SimpleVolumePlugin * reinterpretSimpleVolumePlugin(Plugin * _plugin){
      return (SimpleVolumePlugin *)_plugin;
   }
   
   SimpleVolumePlugin * getSimpleVolumePlugin(){
         return (SimpleVolumePlugin *)Simulator::pluginManager.get("SimpleVolume");
    }

%}

%include <VolumeMean/VolumeMean.h>
// %include <VolumeMean/VolumeMeanParseData.h>

%inline %{
   VolumeMean * reinterpretVolumeMean(Steppable * _steppable){
      return (VolumeMean *)_steppable;
   }

   VolumeMean * getVolumeMeanSteppable(){
         return (VolumeMean *)Simulator::steppableManager.get("VolumeMean");
    }
   
   
%}




// ********************************************* END OF SECTION ********************************** ************************************************




