#ifndef ECMATERIALSSTEPPABLE_EXPORT_H
#define ECMATERIALSSTEPPABLE_EXPORT_H
    #if defined(_WIN32)
      #ifdef ECMaterialsSteppableShared_EXPORTS
          #define ECMATERIALSSTEPPABLE_EXPORT __declspec(dllexport)
          #define ECMATERIALSSTEPPABLE_EXPIMP_TEMPLATE
      #else
          #define ECMATERIALSSTEPPABLE_EXPORT __declspec(dllimport)
          #define ECMATERIALSSTEPPABLE_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define ECMATERIALSSTEPPABLE_EXPORT
         #define ECMATERIALSSTEPPABLE_EXPIMP_TEMPLATE
    #endif
#endif