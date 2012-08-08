#ifndef PIXELTRACKER_EXPORT_H
#define PIXELTRACKER_EXPORT_H

    #if defined(_WIN32)
      #ifdef PixelTrackerShared_EXPORTS
          #define PIXELTRACKER_EXPORT __declspec(dllexport)
          #define PIXELTRACKER_EXPIMP_TEMPLATE
      #else
          #define PIXELTRACKER_EXPORT __declspec(dllimport)
          #define PIXELTRACKER_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define PIXELTRACKER_EXPORT
         #define PIXELTRACKER_EXPIMP_TEMPLATE
    #endif

#endif
