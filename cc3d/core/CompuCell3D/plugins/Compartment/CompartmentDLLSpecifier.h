#ifndef COMPARTMENT_EXPORT_H
#define COMPARTMENT_EXPORT_H

    #if defined(_WIN32)
      #ifdef CompartmentShared_EXPORTS
          #define COMPARTMENT_EXPORT __declspec(dllexport)
          #define COMPARTMENT_EXPIMP_TEMPLATE
      #else
          #define COMPARTMENT_EXPORT __declspec(dllimport)
          #define COMPARTMENT_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define COMPARTMENT_EXPORT
         #define COMPARTMENT_EXPIMP_TEMPLATE
    #endif

#endif
