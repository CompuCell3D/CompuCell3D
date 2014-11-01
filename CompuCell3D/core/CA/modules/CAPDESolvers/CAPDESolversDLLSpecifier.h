#ifndef CAPDESOLVERS_EXPORT_H
#define CAPDESOLVERS_EXPORT_H

    #if defined(_WIN32)
      #ifdef CAPDESolversShared_EXPORTS
          #define CAPDESOLVERS_EXPORT __declspec(dllexport)
          #define CAPDESOLVERS_EXPIMP_TEMPLATE
      #else
          #define CAPDESOLVERS_EXPORT __declspec(dllimport)
          #define CAPDESOLVERS_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CAPDESOLVERS_EXPORT
         #define CAPDESOLVERS_EXPIMP_TEMPLATE
    #endif

#endif
