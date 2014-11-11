#ifndef CELLTRAIL_EXPORT_H
#define CELLTRAIL_EXPORT_H

    #if defined(_WIN32)
      #ifdef CellTrailShared_EXPORTS
          #define CELLTRAIL_EXPORT __declspec(dllexport)
          #define CELLTRAIL_EXPIMP_TEMPLATE
      #else
          #define CELLTRAIL_EXPORT __declspec(dllimport)
          #define CELLTRAIL_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CELLTRAIL_EXPORT
         #define CELLTRAIL_EXPIMP_TEMPLATE
    #endif

#endif
