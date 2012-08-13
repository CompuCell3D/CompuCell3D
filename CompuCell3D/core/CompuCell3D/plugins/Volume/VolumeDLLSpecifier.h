#ifndef VOLUME_EXPORT_H
#define VOLUME_EXPORT_H

    #if defined(_WIN32)
      #ifdef VolumeShared_EXPORTS
          #define VOLUME_EXPORT __declspec(dllexport)
          #define VOLUME_EXPIMP_TEMPLATE
      #else
          #define VOLUME_EXPORT __declspec(dllimport)
          #define VOLUME_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define VOLUME_EXPORT
         #define VOLUME_EXPIMP_TEMPLATE
    #endif

#endif
