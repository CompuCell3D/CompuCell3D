#ifndef SURFACE_EXPORT_H
#define SURFACE_EXPORT_H

    #if defined(_WIN32)
      #ifdef SurfaceShared_EXPORTS
          #define SURFACE_EXPORT __declspec(dllexport)
          #define SURFACE_EXPIMP_TEMPLATE
      #else
          #define SURFACE_EXPORT __declspec(dllimport)
          #define SURFACE_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define SURFACE_EXPORT
         #define SURFACE_EXPIMP_TEMPLATE
    #endif

#endif
