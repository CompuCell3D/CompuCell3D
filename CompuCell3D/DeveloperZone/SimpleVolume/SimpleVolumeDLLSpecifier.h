#ifndef SIMPLEVOLUME_EXPORT_H
#define SIMPLEVOLUME_EXPORT_H

    #if defined(_WIN32)
      #ifdef SimpleVolumeShared_EXPORTS
          #define SIMPLEVOLUME_EXPORT __declspec(dllexport)
          #define SIMPLEVOLUME_EXPIMP_TEMPLATE
      #else
          #define SIMPLEVOLUME_EXPORT __declspec(dllimport)
          #define SIMPLEVOLUME_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define SIMPLEVOLUME_EXPORT
         #define SIMPLEVOLUME_EXPIMP_TEMPLATE
    #endif

#endif
