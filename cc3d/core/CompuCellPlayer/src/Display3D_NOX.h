#ifndef DISPLAY3D_NOX_H
#define DISPLAY3D_NOX_H

#include <Display3DBase.h>




class Display3D_NOX : public Display3DBase
{
   public:
   Display3D_NOX(const char *name = 0);
   ~Display3D_NOX();
   virtual void initializeDisplay3D();

};



#endif
