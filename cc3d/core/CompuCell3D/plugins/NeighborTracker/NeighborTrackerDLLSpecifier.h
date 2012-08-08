#ifndef NEIGHBORTRACKER_EXPORT_H
#define NEIGHBORTRACKER_EXPORT_H

    #if defined(_WIN32)
      #ifdef NeighborTrackerShared_EXPORTS
          #define NEIGHBORTRACKER_EXPORT __declspec(dllexport)
          #define NEIGHBORTRACKER_EXPIMP_TEMPLATE
      #else
          #define NEIGHBORTRACKER_EXPORT __declspec(dllimport)
          #define NEIGHBORTRACKER_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define NEIGHBORTRACKER_EXPORT
         #define NEIGHBORTRACKER_EXPIMP_TEMPLATE
    #endif

#endif
