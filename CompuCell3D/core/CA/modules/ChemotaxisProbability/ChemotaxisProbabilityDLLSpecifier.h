#ifndef CHEMOTAXISPROBABILITY_EXPORTS_H
#define CHEMOTAXISPROBABILITY_EXPORTS_H

    #if defined(_WIN32)
      #ifdef ChemotaxisProbabilityShared_EXPORTS
          #define CHEMOTAXISPROBABILITY_EXPORT __declspec(dllexport)
          #define CHEMOTAXISPROBABILITY_EXPIMP_TEMPLATE
      #else
          #define CHEMOTAXISPROBABILITY_EXPORT __declspec(dllimport)
          #define CHEMOTAXISPROBABILITY_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CHEMOTAXISPROBABILITY_EXPORT
         #define CHEMOTAXISPROBABILITY_EXPIMP_TEMPLATE
    #endif

#endif
