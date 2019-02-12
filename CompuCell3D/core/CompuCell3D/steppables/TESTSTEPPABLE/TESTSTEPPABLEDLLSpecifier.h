
#ifndef TESTSTEPPABLE_EXPORT_H
#define TESTSTEPPABLE_EXPORT_H

    #if defined(_WIN32)
      #ifdef TESTSTEPPABLEShared_EXPORTS
          #define TESTSTEPPABLE_EXPORT __declspec(dllexport)
          #define TESTSTEPPABLE_EXPIMP_TEMPLATE
      #else
          #define TESTSTEPPABLE_EXPORT __declspec(dllimport)
          #define TESTSTEPPABLE_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define TESTSTEPPABLE_EXPORT
         #define TESTSTEPPABLE_EXPIMP_TEMPLATE
    #endif

#endif
