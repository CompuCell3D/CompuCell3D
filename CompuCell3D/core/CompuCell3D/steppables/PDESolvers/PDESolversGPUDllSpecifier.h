#ifndef FLEXDIFF_EXPORT_H
#define FLEXDIFF_EXPORT_H

    #if defined(_WIN32)
      #ifdef PDESOLVERSGPU_EXPORT_COMPILER_DEFINITION
          #define PDESOLVERSGPU_EXPORT __declspec(dllexport)
          #define PDESOLVERSGPU_EXPIMP_TEMPLATE
      #else
          #define PDESOLVERSGPU_EXPORT __declspec(dllimport)
          #define PDESOLVERSGPU_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define PDESOLVERSGPU_EXPORT
         #define PDESOLVERSGPU_EXPIMP_TEMPLATE
    #endif

#endif
