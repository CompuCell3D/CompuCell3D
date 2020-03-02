

#ifndef PRESSURECALCULATORPATA_H

#define PRESSURECALCULATORPATA_H





#include <vector>

#include "PressureCalculatorDLLSpecifier.h"



namespace CompuCell3D {



   

   class PRESSURECALCULATOR_EXPORT PressureCalculatorData{

      public:

         PressureCalculatorData(){};

         

         ~PressureCalculatorData(){};

         std::vector<float> array;

         int x;

         

         

   };

};

#endif

