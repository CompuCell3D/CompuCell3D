#ifndef CELLORIENTATION_EXPORT_H
#define CELLORIENTATION_EXPORT_H

    #if defined(_WIN32)
      #ifdef CellOrientationShared_EXPORTS
          #define CELLORIENTATION_EXPORT __declspec(dllexport)
          #define CELLORIENTATION_EXPIMP_TEMPLATE
      #else
          #define CELLORIENTATION_EXPORT __declspec(dllimport)
          #define CELLORIENTATION_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CELLORIENTATION_EXPORT
         #define CELLORIENTATION_EXPIMP_TEMPLATE
    #endif

#endif
