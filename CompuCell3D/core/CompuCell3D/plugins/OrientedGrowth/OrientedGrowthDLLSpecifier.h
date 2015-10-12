
#ifndef ORIENTEDGROWTH_EXPORT_H
#define ORIENTEDGROWTH_EXPORT_H

    #if defined(_WIN32)
      #ifdef OrientedGrowthShared_EXPORTS
          #define ORIENTEDGROWTH_EXPORT __declspec(dllexport)
          #define ORIENTEDGROWTH_EXPIMP_TEMPLATE
      #else
          #define ORIENTEDGROWTH_EXPORT __declspec(dllimport)
          #define ORIENTEDGROWTH_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define ORIENTEDGROWTH_EXPORT
         #define ORIENTEDGROWTH_EXPIMP_TEMPLATE
    #endif

#endif
