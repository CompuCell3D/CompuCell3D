
#ifndef POLARIZATION23_EXPORT_H
#define POLARIZATION23_EXPORT_H

    #if defined(_WIN32)
      #ifdef Polarization23Shared_EXPORTS
          #define POLARIZATION23_EXPORT __declspec(dllexport)
          #define POLARIZATION23_EXPIMP_TEMPLATE
      #else
          #define POLARIZATION23_EXPORT __declspec(dllimport)
          #define POLARIZATION23_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define POLARIZATION23_EXPORT
         #define POLARIZATION23_EXPIMP_TEMPLATE
    #endif

#endif
