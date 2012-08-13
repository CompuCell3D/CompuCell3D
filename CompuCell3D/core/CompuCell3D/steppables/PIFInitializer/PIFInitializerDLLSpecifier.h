#ifndef PIFINITIALIZER_EXPORT_H
#define PIFINITIALIZER_EXPORT_H

    #if defined(_WIN32)
      #ifdef PIFInitializerShared_EXPORTS
          #define PIFINITIALIZER_EXPORT __declspec(dllexport)
          #define PIFINITIALIZER_EXPIMP_TEMPLATE
      #else
          #define PIFINITIALIZER_EXPORT __declspec(dllimport)
          #define PIFINITIALIZER_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define PIFINITIALIZER_EXPORT
         #define PIFINITIALIZER_EXPIMP_TEMPLATE
    #endif

#endif
