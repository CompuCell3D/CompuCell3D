#ifndef PDESOLVERS_EXPORT_H
#define PDESOLVERS_EXPORT_H

    #if defined(_WIN32)
      #ifdef PDESolversShared_EXPORTS
          #define PDESOLVERS_EXPORT __declspec(dllexport)
          #define PDESOLVERS_EXPIMP_TEMPLATE
      #else
          #define PDESOLVERS_EXPORT __declspec(dllimport)
          #define PDESOLVERS_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define PDESOLVERS_EXPORT
         #define PDESOLVERS_EXPIMP_TEMPLATE
    #endif

#endif

