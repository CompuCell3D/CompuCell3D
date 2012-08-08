#ifndef LENGTHCONSTRAINT_EXPORT_H
#define LENGTHCONSTRAINT_EXPORT_H

    #if defined(_WIN32)
      #ifdef LengthConstraintShared_EXPORTS
          #define LENGTHCONSTRAINT_EXPORT __declspec(dllexport)
          #define LENGTHCONSTRAINT_EXPIMP_TEMPLATE
      #else
          #define LENGTHCONSTRAINT_EXPORT __declspec(dllimport)
          #define LENGTHCONSTRAINT_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define LENGTHCONSTRAINT_EXPORT
         #define LENGTHCONSTRAINT_EXPIMP_TEMPLATE
    #endif

#endif
