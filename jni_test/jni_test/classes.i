
%module classes

%include "typemaps.i"

%include <windows.i>
%{
#include <jni.h>
#include "classes.h"

#include "dllDeclarationSpecifier.h"

#define DECLSPECIFIER //have to include this to avoid problems with interpreting by swig win32 specific c++ extensions...


using namespace std;
using namespace CompuCell3D;

%}



#define DECLSPECIFIER //have to include this to avoid problems with interpreting by swig win32 specific c++ extensions...

%include "classes.h"

using namespace CompuCell3D;
