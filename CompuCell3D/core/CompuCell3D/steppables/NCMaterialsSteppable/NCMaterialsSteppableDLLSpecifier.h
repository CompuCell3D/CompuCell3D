#ifndef NCMATERIALSSTEPPABLE_EXPORT_H
#define NCMATERIALSSTEPPABLE_EXPORT_H
    #if defined(_WIN32)
      #ifdef NCMaterialsSteppableShared_EXPORTS
          #define NCMATERIALSSTEPPABLE_EXPORT __declspec(dllexport)
          #define NCMATERIALSSTEPPABLE_EXPIMP_TEMPLATE
      #else
          #define NCMATERIALSSTEPPABLE_EXPORT __declspec(dllimport)
          #define NCMATERIALSSTEPPABLE_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define NCMATERIALSSTEPPABLE_EXPORT
         #define NCMATERIALSSTEPPABLE_EXPIMP_TEMPLATE
    #endif
#endif