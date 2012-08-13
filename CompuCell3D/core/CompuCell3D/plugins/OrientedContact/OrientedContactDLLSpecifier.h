#ifndef ORIENTEDCONTACT_EXPORT_H
#define ORIENTEDCONTACT_EXPORT_H

    #if defined(_WIN32)
      #ifdef OrientedContactShared_EXPORTS
          #define ORIENTEDCONTACT_EXPORT __declspec(dllexport)
          #define ORIENTEDCONTACT_EXPIMP_TEMPLATE
      #else
          #define ORIENTEDCONTACT_EXPORT __declspec(dllimport)
          #define ORIENTEDCONTACT_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define ORIENTEDCONTACT_EXPORT
         #define ORIENTEDCONTACT_EXPIMP_TEMPLATE
    #endif

#endif
