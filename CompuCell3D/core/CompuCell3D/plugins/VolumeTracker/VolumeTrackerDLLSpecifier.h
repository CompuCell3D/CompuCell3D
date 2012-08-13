#ifndef VOLUMETRACKER_EXPORT_H
#define VOLUMETRACKER_EXPORT_H

    #if defined(_WIN32)
      #ifdef VolumeTrackerShared_EXPORTS
          #define VOLUMETRACKER_EXPORT __declspec(dllexport)
          #define VOLUMETRACKER_EXPIMP_TEMPLATE
      #else
          #define VOLUMETRACKER_EXPORT __declspec(dllimport)
          #define VOLUMETRACKER_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define VOLUMETRACKER_EXPORT
         #define VOLUMETRACKER_EXPIMP_TEMPLATE
    #endif

#endif
