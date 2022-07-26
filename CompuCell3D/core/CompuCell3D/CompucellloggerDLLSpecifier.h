#ifndef CAPI_EXPORT_H
#define CAPI_EXPORT_H

    #if defined(_WIN32)
      #ifdef CAPI_Shared_EXPORTS
          #define CAPI_EXPORT __declspec(dllexport)
          #define CAPI_EXPIMP_TEMPLATE
      #else
          #define CAPI_EXPORT __declspec(dllimport)
          #define CAPI_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CAPI_EXPORT
         #define CAPI_EXPIMP_TEMPLATE
    #endif

#endif

