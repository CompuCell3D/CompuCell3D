#ifndef NCMATERIALS_EXPORT_H
#define NCMATERIALS_EXPORT_H

    #if defined(_WIN32)
      #ifdef NCMaterialsShared_EXPORTS
          #define NCMATERIALS_EXPORT __declspec(dllexport)
          #define NCMATERIALS_EXPIMP_TEMPLATE
      #else
          #define NCMATERIALS_EXPORT __declspec(dllimport)
          #define NCMATERIALS_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define NCMATERIALS_EXPORT
         #define NCMATERIALS_EXPIMP_TEMPLATE
    #endif

#endif
