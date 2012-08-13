#ifndef LENGTHCONSTRAINTLOCALFLEX_EXPORT_H
#define LENGTHCONSTRAINTLOCALFLEX_EXPORT_H

    #if defined(_WIN32)
      #ifdef LengthConstraintLocalFlexShared_EXPORTS
          #define LENGTHCONSTRAINTLOCALFLEX_EXPORT __declspec(dllexport)
          #define LENGTHCONSTRAINTLOCALFLEX_EXPIMP_TEMPLATE
      #else
          #define LENGTHCONSTRAINTLOCALFLEX_EXPORT __declspec(dllimport)
          #define LENGTHCONSTRAINTLOCALFLEX_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define LENGTHCONSTRAINTLOCALFLEX_EXPORT
         #define LENGTHCONSTRAINTLOCALFLEX_EXPIMP_TEMPLATE
    #endif

#endif
