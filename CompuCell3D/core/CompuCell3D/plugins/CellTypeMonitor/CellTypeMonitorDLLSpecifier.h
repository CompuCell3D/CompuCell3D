
#ifndef CELLTYPEMONITOR_EXPORT_H
#define CELLTYPEMONITOR_EXPORT_H

    #if defined(_WIN32)
      #ifdef CellTypeMonitorShared_EXPORTS
          #define CELLTYPEMONITOR_EXPORT __declspec(dllexport)
          #define CELLTYPEMONITOR_EXPIMP_TEMPLATE
      #else
          #define CELLTYPEMONITOR_EXPORT __declspec(dllimport)
          #define CELLTYPEMONITOR_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CELLTYPEMONITOR_EXPORT
         #define CELLTYPEMONITOR_EXPIMP_TEMPLATE
    #endif

#endif
