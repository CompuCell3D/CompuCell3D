#ifndef INTERFACES_EXPORT_H
#define INTERFACES_EXPORT_H

    #if defined(_WIN32)
      #ifdef InterfacesShared_EXPORTS
          #define INTERFACES_EXPORT __declspec(dllexport)
          #define INTERFACES_EXPIMP_TEMPLATE
      #else
          #define INTERFACES_EXPORT __declspec(dllimport)
          #define INTERFACES_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define INTERFACES_EXPORT
         #define INTERFACES_EXPIMP_TEMPLATE
    #endif

#endif
