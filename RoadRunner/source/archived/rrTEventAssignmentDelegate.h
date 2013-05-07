#ifndef rrTEventAssignmentDelegateH
#define rrTEventAssignmentDelegateH

#include "rrExporter.h"

namespace rr
{
    typedef void                         (callConv *TEventAssignmentDelegate)();          //FuncPointer taking no args and returning void
    typedef TEventAssignmentDelegate*    (callConv *c_TEventAssignmentDelegateStar)();    //Array of function pointers
}

#endif

