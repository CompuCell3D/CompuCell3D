#ifndef BOUNDARYPIXELTRACKER_EXPORT_H
#define BOUNDARYPIXELTRACKER_EXPORT_H

    #if defined(_WIN32)
      #ifdef BoundaryPixelTrackerShared_EXPORTS
          #define BOUNDARYPIXELTRACKER_EXPORT __declspec(dllexport)
          #define BOUNDARYPIXELTRACKER_EXPIMP_TEMPLATE
      #else
          #define BOUNDARYPIXELTRACKER_EXPORT __declspec(dllimport)
          #define BOUNDARYPIXELTRACKER_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define BOUNDARYPIXELTRACKER_EXPORT
         #define BOUNDARYPIXELTRACKER_EXPIMP_TEMPLATE
    #endif

#endif
