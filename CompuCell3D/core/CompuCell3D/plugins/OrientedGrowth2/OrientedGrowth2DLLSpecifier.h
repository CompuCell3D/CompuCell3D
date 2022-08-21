
#ifndef ORIENTEDGROWTH2_EXPORT_H
#define ORIENTEDGROWTH2_EXPORT_H

    #if defined(_WIN32)
      #ifdef OrientedGrowth2Shared_EXPORTS
          #define ORIENTEDGROWTH2_EXPORT __declspec(dllexport)
          #define ORIENTEDGROWTH2_EXPIMP_TEMPLATE
      #else
          #define ORIENTEDGROWTH2_EXPORT __declspec(dllimport)
          #define ORIENTEDGROWTH2_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define ORIENTEDGROWTH2_EXPORT
         #define ORIENTEDGROWTH2_EXPIMP_TEMPLATE
    #endif

#endif
