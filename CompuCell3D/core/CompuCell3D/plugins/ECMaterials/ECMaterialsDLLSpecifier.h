#ifndef ECMATERIALS_EXPORT_H
#define ECMATERIALS_EXPORT_H

    #if defined(_WIN32)
      #ifdef ECMaterialsShared_EXPORTS
          #define ECMATERIALS_EXPORT __declspec(dllexport)
          #define ECMATERIALS_EXPIMP_TEMPLATE
      #else
          #define ECMATERIALS_EXPORT __declspec(dllimport)
          #define ECMATERIALS_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define ECMATERIALS_EXPORT
         #define ECMATERIALS_EXPIMP_TEMPLATE
    #endif

#endif
