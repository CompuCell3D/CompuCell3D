#ifndef CELLTYPE_EXPORT_H
#define CELLTYPE_EXPORT_H

    #if defined(_WIN32)
      #ifdef CellTypeShared_EXPORTS
          #define CELLTYPE_EXPORT __declspec(dllexport)
          #define CELLTYPE_EXPIMP_TEMPLATE
      #else
          #define CELLTYPE_EXPORT __declspec(dllimport)
          #define CELLTYPE_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CELLTYPE_EXPORT
         #define CELLTYPE_EXPIMP_TEMPLATE
    #endif

#endif
