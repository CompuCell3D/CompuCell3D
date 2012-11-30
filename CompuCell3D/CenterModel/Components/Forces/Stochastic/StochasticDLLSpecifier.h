#ifndef STOCHASTIC_EXPORT_H
#define STOCHASTIC_EXPORT_H

    #if defined(_WIN32)
      #ifdef StochasticShared_EXPORTS
          #define STOCHASTIC_EXPORT __declspec(dllexport)
          #define STOCHASTIC_EXPIMP_TEMPLATE
      #else
          #define STOCHASTIC_EXPORT __declspec(dllimport)
          #define STOCHASTIC_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define STOCHASTIC_EXPORT
         #define STOCHASTIC_EXPIMP_TEMPLATE
    #endif

#endif
