#ifndef NEIGHBORSTICK_EXPORT_H
#define NEIGHBORSTICK_EXPORT_H

    #if defined(_WIN32)
      #ifdef NeighborStickShared_EXPORTS
          #define NEIGHBORSTICK_EXPORT __declspec(dllexport)
          #define NEIGHBORSTICK_EXPIMP_TEMPLATE
      #else
          #define NEIGHBORSTICK_EXPORT __declspec(dllimport)
          #define NEIGHBORSTICK_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define NEIGHBORSTICK_EXPORT
         #define NEIGHBORSTICK_EXPIMP_TEMPLATE
    #endif

#endif
