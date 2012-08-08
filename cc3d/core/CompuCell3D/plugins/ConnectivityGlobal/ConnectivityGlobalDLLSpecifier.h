#ifndef CONNECTIVITYGLOBAL_EXPORT_H
#define CONNECTIVITYGLOBAL_EXPORT_H

    #if defined(_WIN32)
      #ifdef ConnectivityGlobalShared_EXPORTS
          #define CONNECTIVITYGLOBAL_EXPORT __declspec(dllexport)
          #define CONNECTIVITYGLOBAL_EXPIMP_TEMPLATE
      #else
          #define CONNECTIVITYGLOBAL_EXPORT __declspec(dllimport)
          #define CONNECTIVITYGLOBAL_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CONNECTIVITYGLOBAL_EXPORT
         #define CONNECTIVITYGLOBAL_EXPIMP_TEMPLATE
    #endif

#endif
