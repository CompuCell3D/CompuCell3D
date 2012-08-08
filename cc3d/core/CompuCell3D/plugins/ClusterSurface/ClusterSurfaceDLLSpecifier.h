
#ifndef CLUSTERSURFACE_EXPORT_H
#define CLUSTERSURFACE_EXPORT_H

    #if defined(_WIN32)
      #ifdef ClusterSurfaceShared_EXPORTS
          #define CLUSTERSURFACE_EXPORT __declspec(dllexport)
          #define CLUSTERSURFACE_EXPIMP_TEMPLATE
      #else
          #define CLUSTERSURFACE_EXPORT __declspec(dllimport)
          #define CLUSTERSURFACE_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CLUSTERSURFACE_EXPORT
         #define CLUSTERSURFACE_EXPIMP_TEMPLATE
    #endif

#endif
