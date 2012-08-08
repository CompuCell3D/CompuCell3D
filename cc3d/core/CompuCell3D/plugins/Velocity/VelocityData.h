#ifndef VELOCITYDATA_H
#define VELOCITYDATA_H

#include <Utils/Coordinates3D.h>

namespace CompuCell3D{

class VelocityData{
   public:
      Coordinates3D<float> beforeFlipCM; //this will be updated every spin flip attempt but is overwritten every time we try doing spin
      // flip attempt. This is to make book keeping easy
      Coordinates3D<float> afterFlipCM; //this will be updated every spin flip attempt and is used only to tell user what the 
      //after flip Center of Mass would be

      Coordinates3D<float> velocity;//this will be updated every spin flip
      
      Coordinates3D<float> afterFlipVelocity;//this will be updated every spin flip attempt is used only to tell user what the 
      //after flip velocity would be

      VelocityData(){zeroAll();}

      void zeroAll(){
         Coordinates3D<float> zeroVec(0.,0.,0.);
         beforeFlipCM = zeroVec;
         afterFlipCM = zeroVec;
         velocity = zeroVec;
         afterFlipVelocity = zeroVec;
         
      }

};


};
#endif
