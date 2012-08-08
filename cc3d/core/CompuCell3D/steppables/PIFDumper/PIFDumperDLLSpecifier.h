#ifndef PIFDUMPER_EXPORT_H
#define PIFDUMPER_EXPORT_H

    #if defined(_WIN32)
      #ifdef PIFDumperShared_EXPORTS
          #define PIFDUMPER_EXPORT __declspec(dllexport)
          #define PIFDUMPER_EXPIMP_TEMPLATE
      #else
          #define PIFDUMPER_EXPORT __declspec(dllimport)
          #define PIFDUMPER_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define PIFDUMPER_EXPORT
         #define PIFDUMPER_EXPIMP_TEMPLATE
    #endif

#endif
