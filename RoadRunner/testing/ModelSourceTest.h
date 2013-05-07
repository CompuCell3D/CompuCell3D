#ifndef modelH
#define modelH
#include <stdio.h>
#include <stdbool.h>
#if defined(WIN32)
    #if defined(BUILD_MODEL_DLL)
        #define D_S __declspec(dllexport)
    #else
        #define D_S __declspec(dllimport)
    #endif
#else
    #define D_S 
#endif


D_S int TestFunction();                          



#endif 
