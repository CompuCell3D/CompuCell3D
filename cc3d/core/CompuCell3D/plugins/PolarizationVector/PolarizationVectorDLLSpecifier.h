#ifndef POLARIZATIONVECTOR_EXPORT_H
#define POLARIZATIONVECTOR_EXPORT_H

    #if defined(_WIN32)
      #ifdef PolarizationVectorShared_EXPORTS
          #define POLARIZATIONVECTOR_EXPORT __declspec(dllexport)
          #define POLARIZATIONVECTOR_EXPIMP_TEMPLATE
      #else
          #define POLARIZATIONVECTOR_EXPORT __declspec(dllimport)
          #define POLARIZATIONVECTOR_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define POLARIZATIONVECTOR_EXPORT
         #define POLARIZATIONVECTOR_EXPIMP_TEMPLATE
    #endif

#endif
