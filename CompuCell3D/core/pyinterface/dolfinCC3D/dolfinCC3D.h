#ifndef DOLFINCC3D_H
#define DOLFINCC3D_H


#include "DolfinCC3DDLLSpecifier.h"

   class DOLFINCC3D_EXPORT dolfinCC3D{
      public:
         dolfinCC3D():x(12.00){}
         ~dolfinCC3D();
         double getX();
         double x;
   };



#endif