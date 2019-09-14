#ifndef CUSTOMCELLATTRIBUTESTEPPABLEPATA_H
#define CUSTOMCELLATTRIBUTESTEPPABLEPATA_H

#include <vector>
#include "CustomCellAttributeSteppableDLLSpecifier.h"

namespace CompuCell3D {

   class CUSTOMCELLATTRIBUTESTEPPABLE_EXPORT CustomCellAttributeSteppableData{

      public:

         CustomCellAttributeSteppableData(){};
         ~CustomCellAttributeSteppableData(){};

         std::vector<float> array;

         int x;

   };

};

#endif

