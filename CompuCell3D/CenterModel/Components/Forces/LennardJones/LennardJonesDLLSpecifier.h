#ifndef LENNARDJONES_EXPORT_H
#define LENNARDJONES_EXPORT_H

    #if defined(_WIN32)
      #ifdef LennardJonesShared_EXPORTS
          #define LENNARDJONES_EXPORT __declspec(dllexport)
          #define LENNARDJONES_EXPIMP_TEMPLATE
      #else
          #define LENNARDJONES_EXPORT __declspec(dllimport)
          #define LENNARDJONES_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define LENNARDJONES_EXPORT
         #define LENNARDJONES_EXPIMP_TEMPLATE
    #endif

#endif
