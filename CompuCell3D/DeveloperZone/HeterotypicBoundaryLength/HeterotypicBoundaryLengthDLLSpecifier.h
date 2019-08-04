

#ifndef HETEROTYPICBOUNDARYLENGTH_EXPORT_H

#define HETEROTYPICBOUNDARYLENGTH_EXPORT_H



    #if defined(_WIN32)

      #ifdef HeterotypicBoundaryLengthShared_EXPORTS

          #define HETEROTYPICBOUNDARYLENGTH_EXPORT __declspec(dllexport)

          #define HETEROTYPICBOUNDARYLENGTH_EXPIMP_TEMPLATE

      #else

          #define HETEROTYPICBOUNDARYLENGTH_EXPORT __declspec(dllimport)

          #define HETEROTYPICBOUNDARYLENGTH_EXPIMP_TEMPLATE extern

      #endif

    #else

         #define HETEROTYPICBOUNDARYLENGTH_EXPORT

         #define HETEROTYPICBOUNDARYLENGTH_EXPIMP_TEMPLATE

    #endif



#endif

