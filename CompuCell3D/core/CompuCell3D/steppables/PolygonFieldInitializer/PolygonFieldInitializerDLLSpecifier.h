

#ifndef POLYGONFIELDINITIALIZER_EXPORT_H
#define POLYGONFIELDINITIALIZER_EXPORT_H

    #if defined(_WIN32)

      #ifdef PolygonFieldInitializerShared_EXPORTS

          #define POLYGONFIELDINITIALIZER_EXPORT __declspec(dllexport)

          #define POLYGONFIELDINITIALIZER_EXPIMP_TEMPLATE

      #else

          #define POLYGONFIELDINITIALIZER_EXPORT __declspec(dllimport)

          #define POLYGONFIELDINITIALIZER_EXPIMP_TEMPLATE extern

      #endif

    #else

         #define POLYGONFIELDINITIALIZER_EXPORT

         #define POLYGONFIELDINITIALIZER_EXPIMP_TEMPLATE

    #endif

#endif

