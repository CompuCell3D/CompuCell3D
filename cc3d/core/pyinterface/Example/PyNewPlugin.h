#ifndef PYNEWPLUGIN_H
#define PYNEWPLUGIN_H

#include <CompuCell3D/Plugin.h>
#include "ExampleClassDLLSpecifier.h"

   class EXAMPLECLASS_EXPORT PyNewPlugin{
      public:
         PyNewPlugin():x(11.00){}
         ~PyNewPlugin();
         double getX();
         double x;
   };



#endif