#ifndef CONNECTIVITY_EXPORT_H
#define CONNECTIVITY_EXPORT_H

    #if defined(_WIN32)
      #ifdef ConnectivityShared_EXPORTS
          #define CONNECTIVITY_EXPORT __declspec(dllexport)
          #define CONNECTIVITY_EXPIMP_TEMPLATE
      #else
          #define CONNECTIVITY_EXPORT __declspec(dllimport)
          #define CONNECTIVITY_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CONNECTIVITY_EXPORT
         #define CONNECTIVITY_EXPIMP_TEMPLATE
    #endif

#endif
