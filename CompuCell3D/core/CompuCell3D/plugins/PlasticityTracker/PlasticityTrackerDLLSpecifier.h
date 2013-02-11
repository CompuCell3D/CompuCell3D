#ifndef PLASTICITYTRACKER_EXPORT_H
#define PLASTICITYTRACKER_EXPORT_H

    #if defined(_WIN32)
      #ifdef PlasticityTrackerShared_EXPORTS
          #define PLASTICITYTRACKER_EXPORT __declspec(dllexport)
          #define PLASTICITYTRACKER_EXPIMP_TEMPLATE
      #else
          #define PLASTICITYTRACKER_EXPORT __declspec(dllimport)
          #define PLASTICITYTRACKER_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define PLASTICITYTRACKER_EXPORT
         #define PLASTICITYTRACKER_EXPIMP_TEMPLATE
    #endif

#endif
