

#ifndef PRESSURECALCULATOR_EXPORT_H

#define PRESSURECALCULATOR_EXPORT_H



    #if defined(_WIN32)

      #ifdef PressureCalculatorShared_EXPORTS

          #define PRESSURECALCULATOR_EXPORT __declspec(dllexport)

          #define PRESSURECALCULATOR_EXPIMP_TEMPLATE

      #else

          #define PRESSURECALCULATOR_EXPORT __declspec(dllimport)

          #define PRESSURECALCULATOR_EXPIMP_TEMPLATE extern

      #endif

    #else

         #define PRESSURECALCULATOR_EXPORT

         #define PRESSURECALCULATOR_EXPIMP_TEMPLATE

    #endif



#endif

