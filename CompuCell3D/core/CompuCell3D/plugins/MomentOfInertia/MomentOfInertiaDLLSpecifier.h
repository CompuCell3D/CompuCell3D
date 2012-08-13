#ifndef MOMENTOFINERTIA_EXPORT_H
#define MOMENTOFINERTIA_EXPORT_H

    #if defined(_WIN32)
      #ifdef MomentOfInertiaShared_EXPORTS
          #define MOMENTOFINERTIA_EXPORT __declspec(dllexport)
          #define MOMENTOFINERTIA_EXPIMP_TEMPLATE
      #else
          #define MOMENTOFINERTIA_EXPORT __declspec(dllimport)
          #define MOMENTOFINERTIA_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define MOMENTOFINERTIA_EXPORT
         #define MOMENTOFINERTIA_EXPIMP_TEMPLATE
    #endif

#endif
