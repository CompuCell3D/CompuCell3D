

#ifndef GROWTHSTEPPABLE_EXPORT_H

#define GROWTHSTEPPABLE_EXPORT_H



    #if defined(_WIN32)

      #ifdef GrowthSteppableShared_EXPORTS

          #define GROWTHSTEPPABLE_EXPORT __declspec(dllexport)

          #define GROWTHSTEPPABLE_EXPIMP_TEMPLATE

      #else

          #define GROWTHSTEPPABLE_EXPORT __declspec(dllimport)

          #define GROWTHSTEPPABLE_EXPIMP_TEMPLATE extern

      #endif

    #else

         #define GROWTHSTEPPABLE_EXPORT

         #define GROWTHSTEPPABLE_EXPIMP_TEMPLATE

    #endif



#endif

