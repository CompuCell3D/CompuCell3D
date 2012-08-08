#ifndef VOLUMEMEAN_EXPORT_H
#define VOLUMEMEAN_EXPORT_H

    #if defined(_WIN32)
      #ifdef VolumeMeanShared_EXPORTS
          #define VOLUMEMEAN_EXPORT __declspec(dllexport)
          #define VOLUMEMEAN_EXPIMP_TEMPLATE
      #else
          #define VOLUMEMEAN_EXPORT __declspec(dllimport)
          #define VOLUMEMEAN_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define VOLUMEMEAN_EXPORT
         #define VOLUMEMEAN_EXPIMP_TEMPLATE
    #endif

#endif
