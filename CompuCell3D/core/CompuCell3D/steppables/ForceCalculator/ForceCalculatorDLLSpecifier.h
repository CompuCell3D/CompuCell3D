

#ifndef FORCECALCULATOR_EXPORT_H

#define FORCECALCULATOR_EXPORT_H



    #if defined(_WIN32)

      #ifdef ForceCalculatorShared_EXPORTS

          #define FORCECALCULATOR_EXPORT __declspec(dllexport)

          #define FORCECALCULATOR_EXPIMP_TEMPLATE

      #else

          #define FORCECALCULATOR_EXPORT __declspec(dllimport)

          #define FORCECALCULATOR_EXPIMP_TEMPLATE extern

      #endif

    #else

         #define FORCECALCULATOR_EXPORT

         #define FORCECALCULATOR_EXPIMP_TEMPLATE

    #endif



#endif

