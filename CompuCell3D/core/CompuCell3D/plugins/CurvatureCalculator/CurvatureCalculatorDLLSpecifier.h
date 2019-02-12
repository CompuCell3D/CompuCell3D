
#ifndef CURVATURECALCULATOR_EXPORT_H
#define CURVATURECALCULATOR_EXPORT_H

    #if defined(_WIN32)
      #ifdef CurvatureCalculatorShared_EXPORTS
          #define CURVATURECALCULATOR_EXPORT __declspec(dllexport)
          #define CURVATURECALCULATOR_EXPIMP_TEMPLATE
      #else
          #define CURVATURECALCULATOR_EXPORT __declspec(dllimport)
          #define CURVATURECALCULATOR_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CURVATURECALCULATOR_EXPORT
         #define CURVATURECALCULATOR_EXPIMP_TEMPLATE
    #endif

#endif
