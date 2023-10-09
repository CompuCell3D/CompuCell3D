

#ifndef TUBEFIELDINITIALIZER_EXPORT_H
#define TUBEFIELDINITIALIZER_EXPORT_H

    #if defined(_WIN32)

      #ifdef TubeFieldInitializerShared_EXPORTS

          #define TUBEFIELDINITIALIZER_EXPORT __declspec(dllexport)

          #define TUBEFIELDINITIALIZER_EXPIMP_TEMPLATE

      #else

          #define TUBEFIELDINITIALIZER_EXPORT __declspec(dllimport)

          #define TUBEFIELDINITIALIZER_EXPIMP_TEMPLATE extern

      #endif

    #else

         #define TUBEFIELDINITIALIZER_EXPORT

         #define TUBEFIELDINITIALIZER_EXPIMP_TEMPLATE

    #endif

#endif

