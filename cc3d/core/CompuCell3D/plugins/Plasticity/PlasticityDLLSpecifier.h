#ifndef PLASTICITY_EXPORT_H
#define PLASTICITY_EXPORT_H

    #if defined(_WIN32)
      #ifdef PlasticityShared_EXPORTS
          #define PLASTICITY_EXPORT __declspec(dllexport)
          #define PLASTICITY_EXPIMP_TEMPLATE
      #else
          #define PLASTICITY_EXPORT __declspec(dllimport)
          #define PLASTICITY_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define PLASTICITY_EXPORT
         #define PLASTICITY_EXPIMP_TEMPLATE
    #endif

#endif
