#ifndef ELASTICITY_EXPORT_H
#define ELASTICITY_EXPORT_H

    #if defined(_WIN32)
      #ifdef ElasticityShared_EXPORTS
          #define ELASTICITY_EXPORT __declspec(dllexport)
          #define ELASTICITY_EXPIMP_TEMPLATE
      #else
          #define ELASTICITY_EXPORT __declspec(dllimport)
          #define ELASTICITY_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define ELASTICITY_EXPORT
         #define ELASTICITY_EXPIMP_TEMPLATE
    #endif

#endif
