
#ifndef CLUSTERSURFACETRACKER_EXPORT_H
#define CLUSTERSURFACETRACKER_EXPORT_H

    #if defined(_WIN32)
      #ifdef ClusterSurfaceTrackerShared_EXPORTS
          #define CLUSTERSURFACETRACKER_EXPORT __declspec(dllexport)
          #define CLUSTERSURFACETRACKER_EXPIMP_TEMPLATE
      #else
          #define CLUSTERSURFACETRACKER_EXPORT __declspec(dllimport)
          #define CLUSTERSURFACETRACKER_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CLUSTERSURFACETRACKER_EXPORT
         #define CLUSTERSURFACETRACKER_EXPIMP_TEMPLATE
    #endif

#endif
