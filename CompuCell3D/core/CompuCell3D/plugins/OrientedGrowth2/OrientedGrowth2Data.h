
#ifndef ORIENTEDGROWTH2DATA_H
#define ORIENTEDGROWTH2DATA_H


#include <vector>
#include "OrientedGrowth2DLLSpecifier.h"

namespace CompuCell3D {

   
   class ORIENTEDGROWTH2_EXPORT OrientedGrowth2Data{
      public:
         OrientedGrowth2Data(){};
         
         ~OrientedGrowth2Data(){};
         std::vector<float> array;
         int x;
         int elong_volume;
         float elong_x;
         float elong_y;
         float elong_z;
         float elong_xCOM;
         float elong_yCOM;
         float elong_zCOM;
         float elong_targetWidth;
         float elong_targetLength;
         float elong_apicalRadius;
         float elong_basalRadius;
         bool elong_enabled;
         bool elong_constricted;
   };
};
#endif
