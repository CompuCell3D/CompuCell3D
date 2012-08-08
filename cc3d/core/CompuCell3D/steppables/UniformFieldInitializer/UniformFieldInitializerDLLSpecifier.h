#ifndef UNIFORMFIELDINITIALIZER_EXPORT_H
#define UNIFORMFIELDINITIALIZER_EXPORT_H

    #if defined(_WIN32)
      #ifdef UniformFieldInitializerShared_EXPORTS
          #define UNIFORMFIELDINITIALIZER_EXPORT __declspec(dllexport)
          #define UNIFORMFIELDINITIALIZER_EXPIMP_TEMPLATE
      #else
          #define UNIFORMFIELDINITIALIZER_EXPORT __declspec(dllimport)
          #define UNIFORMFIELDINITIALIZER_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define UNIFORMFIELDINITIALIZER_EXPORT
         #define UNIFORMFIELDINITIALIZER_EXPIMP_TEMPLATE
    #endif

#endif
