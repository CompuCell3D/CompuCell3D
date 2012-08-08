#ifndef CONTACT_EXPORT_H
#define CONTACT_EXPORT_H

    #if defined(_WIN32)
      #ifdef ContactShared_EXPORTS
          #define CONTACT_EXPORT __declspec(dllexport)
          #define CONTACT_EXPIMP_TEMPLATE
      #else
          #define CONTACT_EXPORT __declspec(dllimport)
          #define CONTACT_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CONTACT_EXPORT
         #define CONTACT_EXPIMP_TEMPLATE
    #endif

#endif
