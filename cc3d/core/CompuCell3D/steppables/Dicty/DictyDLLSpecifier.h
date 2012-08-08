#ifndef DICTY_EXPORT_H
#define DICTY_EXPORT_H

    #if defined(_WIN32)
      #ifdef DictyShared_EXPORTS
          #define DICTY_EXPORT __declspec(dllexport)
          #define DICTY_EXPIMP_TEMPLATE
      #else
          #define DICTY_EXPORT __declspec(dllimport)
          #define DICTY_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define DICTY_EXPORT
         #define DICTY_EXPIMP_TEMPLATE
    #endif

#endif
