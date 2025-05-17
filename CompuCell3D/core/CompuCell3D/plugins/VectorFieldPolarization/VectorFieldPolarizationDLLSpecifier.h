

#ifndef VECTORFIELDPOLARIZATION_EXPORT_H

#define VECTORFIELDPOLARIZATION_EXPORT_H

    #if defined(_WIN32)

      #ifdef VectorFieldPolarizationShared_EXPORTS

          #define VECTORFIELDPOLARIZATION_EXPORT __declspec(dllexport)

          #define VECTORFIELDPOLARIZATION_EXPIMP_TEMPLATE

      #else

          #define VECTORFIELDPOLARIZATION_EXPORT __declspec(dllimport)

          #define VECTORFIELDPOLARIZATION_EXPIMP_TEMPLATE extern

      #endif

    #else

         #define VECTORFIELDPOLARIZATION_EXPORT

         #define VECTORFIELDPOLARIZATION_EXPIMP_TEMPLATE

    #endif

#endif

