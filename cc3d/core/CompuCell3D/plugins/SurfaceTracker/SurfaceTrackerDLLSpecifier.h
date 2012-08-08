#ifndef SURFACETRACKER_EXPORT_H
#define SURFACETRACKER_EXPORT_H

    #if defined(_WIN32)
      #ifdef SurfaceTrackerShared_EXPORTS
          #define SURFACETRACKER_EXPORT __declspec(dllexport)
          #define SURFACETRACKER_EXPIMP_TEMPLATE
      #else
          #define SURFACETRACKER_EXPORT __declspec(dllimport)
          #define SURFACETRACKER_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define SURFACETRACKER_EXPORT
         #define SURFACETRACKER_EXPIMP_TEMPLATE
    #endif

#endif
