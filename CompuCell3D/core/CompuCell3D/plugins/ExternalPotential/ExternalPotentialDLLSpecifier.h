#ifndef EXTERNALPOTENTIAL_EXPORT_H
#define EXTERNALPOTENTIAL_EXPORT_H

    #if defined(_WIN32)
      #ifdef ExternalPotentialShared_EXPORTS
          #define EXTERNALPOTENTIAL_EXPORT __declspec(dllexport)
          #define EXTERNALPOTENTIAL_EXPIMP_TEMPLATE
      #else
          #define EXTERNALPOTENTIAL_EXPORT __declspec(dllimport)
          #define EXTERNALPOTENTIAL_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define EXTERNALPOTENTIAL_EXPORT
         #define EXTERNALPOTENTIAL_EXPIMP_TEMPLATE
    #endif

#endif
