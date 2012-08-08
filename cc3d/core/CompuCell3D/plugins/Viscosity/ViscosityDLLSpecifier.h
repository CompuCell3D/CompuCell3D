#ifndef VISCOSITY_EXPORT_H
#define VISCOSITY_EXPORT_H

    #if defined(_WIN32)
      #ifdef ViscosityShared_EXPORTS
          #define VISCOSITY_EXPORT __declspec(dllexport)
          #define VISCOSITY_EXPIMP_TEMPLATE
      #else
          #define VISCOSITY_EXPORT __declspec(dllimport)
          #define VISCOSITY_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define VISCOSITY_EXPORT
         #define VISCOSITY_EXPIMP_TEMPLATE
    #endif

#endif
