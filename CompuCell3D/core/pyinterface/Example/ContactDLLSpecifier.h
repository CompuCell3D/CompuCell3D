#ifndef EXAMPLECLASS_EXPORT_H
#define EXAMPLECLASS_EXPORT_H

    #if defined(_WIN32)
      #ifdef ExampleClass_EXPORTS
          #define EXAMPLECLASS_EXPORT __declspec(dllexport)
          #define EXAMPLECLASS_EXPIMP_TEMPLATE
      #else
          #define EXAMPLECLASS_EXPORT __declspec(dllimport)
          #define EXAMPLECLASS_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define EXAMPLECLASS_EXPORT
         #define EXAMPLECLASS_EXPIMP_TEMPLATE
    #endif

#endif
