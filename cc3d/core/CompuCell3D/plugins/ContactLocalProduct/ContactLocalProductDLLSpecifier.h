#ifndef CONTACTLOCALPRODUCT_EXPORT_H
#define CONTACTLOCALPRODUCT_EXPORT_H

    #if defined(_WIN32)
      #ifdef ContactLocalProductShared_EXPORTS
          #define CONTACTLOCALPRODUCT_EXPORT __declspec(dllexport)
          #define CONTACTLOCALPRODUCT_EXPIMP_TEMPLATE
      #else
          #define CONTACTLOCALPRODUCT_EXPORT __declspec(dllimport)
          #define CONTACTLOCALPRODUCT_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CONTACTLOCALPRODUCT_EXPORT
         #define CONTACTLOCALPRODUCT_EXPIMP_TEMPLATE
    #endif

#endif
