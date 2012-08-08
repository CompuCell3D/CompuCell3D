#ifndef OBJINITIALIZER_EXPORT_H
#define OBJINITIALIZER_EXPORT_H

    #if defined(_WIN32)

      #ifdef ObjInitializerShared_EXPORTS
          #define OBJINITIALIZER_EXPORT __declspec(dllexport)
          #define OBJINITIALIZER_EXPIMP_TEMPLATE
      #else
          #define OBJINITIALIZER_EXPORT __declspec(dllimport)
          #define OBJINITIALIZER_EXPIMP_TEMPLATE extern
      #endif

    #else

         #define OBJINITIALIZER_EXPORT
         #define OBJINITIALIZER_EXPIMP_TEMPLATE

    #endif

#endif
