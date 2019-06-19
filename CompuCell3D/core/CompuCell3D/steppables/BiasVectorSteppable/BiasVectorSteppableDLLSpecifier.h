

#ifndef BIASVECTORSTEPPABLE_EXPORT_H

#define BIASVECTORSTEPPABLE_EXPORT_H



    #if defined(_WIN32)

      #ifdef BiasVectorSteppableShared_EXPORTS

          #define BIASVECTORSTEPPABLE_EXPORT __declspec(dllexport)

          #define BIASVECTORSTEPPABLE_EXPIMP_TEMPLATE

      #else

          #define BIASVECTORSTEPPABLE_EXPORT __declspec(dllimport)

          #define BIASVECTORSTEPPABLE_EXPIMP_TEMPLATE extern

      #endif

    #else

         #define BIASVECTORSTEPPABLE_EXPORT

         #define BIASVECTORSTEPPABLE_EXPIMP_TEMPLATE

    #endif



#endif

