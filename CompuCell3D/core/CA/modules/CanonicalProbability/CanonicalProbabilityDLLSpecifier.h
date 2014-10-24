#ifndef CANONICALPROBABILITY_EXPORTS_H
#define CANONICALPROBABILITY_EXPORTS_H

    #if defined(_WIN32)
      #ifdef CanonicalProbabilityShared_EXPORTS
          #define CANONICALPROBABILITY_EXPORT __declspec(dllexport)
          #define CANONICALPROBABILITY_EXPIMP_TEMPLATE
      #else
          #define CANONICALPROBABILITY_EXPORT __declspec(dllimport)
          #define CANONICALPROBABILITY_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CANONICALPROBABILITY_EXPORT
         #define CANONICALPROBABILITY_EXPIMP_TEMPLATE
    #endif

#endif
