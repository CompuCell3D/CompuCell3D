#ifndef CELLTYPEG_H
#define CELLTYPEG_H
#include <BasicUtils/BasicClassGroup.h>   //had to include it to avoid problems with template instantiation
                                          //obj size for BAsicClassBoundary must be known at the time of instantiation
                                          //it is not enough to use forward declaratin only
/**
@author m
*/


namespace CompuCell3D {
   
   class CellTypeG{
      public:
         CellTypeG():type(0){}
         unsigned char type;         
         
   
   };
};
#endif

