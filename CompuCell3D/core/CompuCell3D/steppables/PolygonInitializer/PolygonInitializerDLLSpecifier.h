

#ifndef POLYGONINITIALIZER_EXPORT_H
#define POLYGONINITIALIZER_EXPORT_H

    #if defined(_WIN32)

      #ifdef PolygonInitializerShared_EXPORTS

          #define POLYGONINITIALIZER_EXPORT __declspec(dllexport)

          #define POLYGONINITIALIZER_EXPIMP_TEMPLATE

      #else

          #define POLYGONINITIALIZER_EXPORT __declspec(dllimport)

          #define POLYGONINITIALIZER_EXPIMP_TEMPLATE extern

      #endif

    #else

         #define POLYGONINITIALIZER_EXPORT

         #define POLYGONINITIALIZER_EXPIMP_TEMPLATE

    #endif

#endif

