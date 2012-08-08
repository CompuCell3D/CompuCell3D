#ifndef SIMTHREADACCESSOR_H
#define SIMTHREADACCESSOR_H
#include "simthreadDLLSpecifier.h"

class SimthreadBase;
extern SimthreadBase SIMTHREAD_EXPORT  * simthreadBasePtr;
extern double SIMTHREAD_EXPORT  numberGlobal;
extern void SIMTHREAD_EXPORT  printLibName();


//SimthreadBase *simthreadBasePtr;
SimthreadBase SIMTHREAD_EXPORT  * getSimthreadBasePtr();
double SIMTHREAD_EXPORT  getNumberGlobal();

#endif
