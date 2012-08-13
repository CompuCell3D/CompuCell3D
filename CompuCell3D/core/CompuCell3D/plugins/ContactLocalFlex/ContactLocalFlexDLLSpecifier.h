#ifndef CONTACTLOCALFLEX_EXPORT_H
#define CONTACTLOCALFLEX_EXPORT_H

    #if defined(_WIN32)
      #ifdef ContactLocalFlexShared_EXPORTS
          #define CONTACTLOCALFLEX_EXPORT __declspec(dllexport)
          #define CONTACTLOCALFLEX_EXPIMP_TEMPLATE
      #else
          #define CONTACTLOCALFLEX_EXPORT __declspec(dllimport)
          #define CONTACTLOCALFLEX_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CONTACTLOCALFLEX_EXPORT
         #define CONTACTLOCALFLEX_EXPIMP_TEMPLATE
    #endif

#endif
