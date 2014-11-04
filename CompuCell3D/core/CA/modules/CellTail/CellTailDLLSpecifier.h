#ifndef CELLTAIL_EXPORT_H
#define CELLTAIL_EXPORT_H

    #if defined(_WIN32)
      #ifdef CellTailShared_EXPORTS
          #define CELLTAIL_EXPORT __declspec(dllexport)
          #define CELLTAIL_EXPIMP_TEMPLATE
      #else
          #define CELLTAIL_EXPORT __declspec(dllimport)
          #define CELLTAIL_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CELLTAIL_EXPORT
         #define CELLTAIL_EXPIMP_TEMPLATE
    #endif

#endif
