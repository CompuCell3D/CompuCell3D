#ifndef FOCALPOINTCONTACT_EXPORT_H
#define FOCALPOINTCONTACT_EXPORT_H

    #if defined(_WIN32)
      #ifdef FocalPointContactShared_EXPORTS
          #define FOCALPOINTCONTACT_EXPORT __declspec(dllexport)
          #define FOCALPOINTCONTACT_EXPIMP_TEMPLATE
      #else
          #define FOCALPOINTCONTACT_EXPORT __declspec(dllimport)
          #define FOCALPOINTCONTACT_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define FOCALPOINTCONTACT_EXPORT
         #define FOCALPOINTCONTACT_EXPIMP_TEMPLATE
    #endif

#endif
