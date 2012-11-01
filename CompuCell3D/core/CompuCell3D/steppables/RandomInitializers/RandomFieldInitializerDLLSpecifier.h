#ifndef RANDOMINITIALIZERS_EXPORT_H
#define RANDOMINITIALIZERS_EXPORT_H

    #if defined(_WIN32)
      #ifdef RandomInitializersShared_EXPORTS
          #define RANDOMINITIALIZERS_EXPORT __declspec(dllexport)
          #define RANDOMINITIALIZERS_EXPIMP_TEMPLATE
      #else
          #define RANDOMINITIALIZERS_EXPORT __declspec(dllimport)
          #define RANDOMINITIALIZERS_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define RANDOMINITIALIZERS_EXPORT
         #define RANDOMINITIALIZERS_EXPIMP_TEMPLATE
    #endif

#endif
