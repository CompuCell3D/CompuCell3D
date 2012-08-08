#ifndef COMPUCELL3DSIMPLECLOCK_H
#define COMPUCELL3DSIMPLECLOCK_H

#include "SimpleClockDLLSpecifier.h"

namespace CompuCell3D {

/**
@author m
*/
class SIMPLECLOCK_EXPORT SimpleClock{
public:
    SimpleClock():clock(0),flag(0)
    {
      
    };
   void decrementUntilZero(){
      if(clock>=1)
         --clock;
   } 
   int clock;
   char flag;
};

};

#endif
