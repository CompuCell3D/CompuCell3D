#ifndef GLOBALBOUNDARYPIXELTRACKER_EXPORT_H
#define GLOBALBOUNDARYPIXELTRACKER_EXPORT_H

    #if defined(_WIN32)
      #ifdef GlobalBoundaryPixelTrackerShared_EXPORTS
          #define GLOBALBOUNDARYPIXELTRACKER_EXPORT __declspec(dllexport)
          #define GLOBALBOUNDARYPIXELTRACKER_EXPIMP_TEMPLATE
      #else
          #define GLOBALBOUNDARYPIXELTRACKER_EXPORT __declspec(dllimport)
          #define GLOBALBOUNDARYPIXELTRACKER_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define GLOBALBOUNDARYPIXELTRACKER_EXPORT
         #define GLOBALBOUNDARYPIXELTRACKER_EXPIMP_TEMPLATE
    #endif

#endif
