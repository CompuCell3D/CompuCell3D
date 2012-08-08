#ifndef CONNECTIVITYLOCALFLEX_EXPORT_H
#define CONNECTIVITYLOCALFLEX_EXPORT_H

    #if defined(_WIN32)
      #ifdef ConnectivityLocalFlexShared_EXPORTS
          #define CONNECTIVITYLOCALFLEX_EXPORT __declspec(dllexport)
          #define CONNECTIVITYLOCALFLEX_EXPIMP_TEMPLATE
      #else
          #define CONNECTIVITYLOCALFLEX_EXPORT __declspec(dllimport)
          #define CONNECTIVITYLOCALFLEX_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CONNECTIVITYLOCALFLEX_EXPORT
         #define CONNECTIVITYLOCALFLEX_EXPIMP_TEMPLATE
    #endif

#endif
