

#ifndef IMPLICITMOTILITY_EXPORT_H

#define IMPLICITMOTILITY_EXPORT_H



    #if defined(_WIN32)

      #ifdef ImplicitMotilityShared_EXPORTS

          #define IMPLICITMOTILITY_EXPORT __declspec(dllexport)

          #define IMPLICITMOTILITY_EXPIMP_TEMPLATE

      #else

          #define IMPLICITMOTILITY_EXPORT __declspec(dllimport)

          #define IMPLICITMOTILITY_EXPIMP_TEMPLATE extern

      #endif

    #else

         #define IMPLICITMOTILITY_EXPORT

         #define IMPLICITMOTILITY_EXPIMP_TEMPLATE

    #endif



#endif

