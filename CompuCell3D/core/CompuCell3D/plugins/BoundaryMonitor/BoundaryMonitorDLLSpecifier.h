
#ifndef BOUNDARYMONITOR_EXPORT_H
#define BOUNDARYMONITOR_EXPORT_H

    #if defined(_WIN32)
      #ifdef BoundaryMonitorShared_EXPORTS
          #define BOUNDARYMONITOR_EXPORT __declspec(dllexport)
          #define BOUNDARYMONITOR_EXPIMP_TEMPLATE
      #else
          #define BOUNDARYMONITOR_EXPORT __declspec(dllimport)
          #define BOUNDARYMONITOR_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define BOUNDARYMONITOR_EXPORT
         #define BOUNDARYMONITOR_EXPIMP_TEMPLATE
    #endif

#endif
